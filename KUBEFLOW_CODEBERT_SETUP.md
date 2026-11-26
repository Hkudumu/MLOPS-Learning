# Fine-tuning CodeBERT with Kubeflow Training Operator + Serving with KServe

End-to-end setup: install the Training Operator, run `train_codebert.py` as a
`PyTorchJob` on the GPU node (`usernode03`), then serve the resulting checkpoint
with KServe.

Cluster facts assumed from this session:
- Context: `LMAPAZ1AKSNPDAI-admin`
- Namespaces in use: `ns-kserve` (KServe controller), `ns-models` (InferenceServices)
- GPU node: `usernode03`, 1 GPU, NVIDIA device plugin already installed
- ACR already attached to AKS (`az aks update --attach-acr`) — no imagePullSecret needed
- Existing fine-tuned model: `codebert-7.0/` (RoBERTa/CodeBERT, binary classifier,
  0=Human, 1=AI), currently served by a custom FastAPI app (`app.py`), not yet via KServe
- `kubectl`/`helm` must be run by you — this session's `kubectl` can't reach the
  API server through the local Netskope TLS-intercepting proxy

---

## Part A0 — Install KServe (cert-manager + CRDs + controller + runtimes)

This is the KServe install already applied to this cluster (`ns-cert-manager`,
`ns-kserve`) — included here for completeness so this doc is a full from-scratch
setup record.

```bash
# 1. cert-manager (required by KServe for webhook TLS certs)
helm upgrade --install cert-manager oci://quay.io/jetstack/charts/cert-manager \
  --version "v1.21.3" \
  --namespace ns-cert-manager \
  --set crds.enabled=true \
  --wait --timeout 5m

# 2. KServe CRDs
helm upgrade --install kserve-crd oci://ghcr.io/kserve/charts/kserve-crd \
  --version "v0.21.0" \
  --namespace ns-kserve \
  --wait --timeout 10m

# 3. KServe controller (Standard deployment mode, using cert-manager for certs)
helm upgrade --install kserve-resources oci://ghcr.io/kserve/charts/kserve-resources \
  --version "v0.21.0" \
  --namespace ns-kserve \
  --set kserve.controller.deploymentMode=Standard \
  --set kserve.certManager.enabled=true \
  --wait --timeout 10m

# 4. Built-in ServingRuntimes (huggingface, sklearn, xgboost, etc.)
helm upgrade --install kserve-runtime-configs oci://ghcr.io/kserve/charts/kserve-runtime-configs \
  --version v0.21.0 \
  --namespace ns-kserve \
  --reuse-values \
  --set kserve.servingruntime.enabled=true \
  --wait
```

Verify:

```bash
kubectl get pods -n ns-cert-manager
kubectl get pods -n ns-kserve
kubectl get clusterservingruntimes -A    # should list huggingface-server-* runtimes
kubectl get crd | grep serving.kserve.io
```

All pods in `ns-cert-manager` and `ns-kserve` should show `Running`, and
`clusterservingruntimes` should include `kserve-huggingfaceserver` — this is
the runtime referenced later in Part G (`modelFormat.name: huggingface`).

---

## Part A — Install Kubeflow Training Operator

The Training Operator is a standalone component (not the full Kubeflow platform).
It adds the `PyTorchJob` CRD and a controller that watches for it.

```bash
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.8.1"

# Verify
kubectl get pods -n kubeflow
kubectl get crd | grep pytorchjobs
```

You should see a `training-operator-xxxxx` pod `Running` in the `kubeflow`
namespace, and `pytorchjobs.kubeflow.org` in the CRD list.

Confirm the GPU node is visible and schedulable:

```bash
kubectl get nodes -L agentpool,kubernetes.azure.com/accelerator
kubectl describe node usernode03 | grep -A5 "Capacity:\|Allocatable:"
```

Note the exact label key/value AKS uses for `usernode03`'s pool (commonly
`agentpool=usernode03` or a custom label) — you'll use it in the `nodeSelector`
below. Also confirm `nvidia.com/gpu: 1` shows up under `Allocatable`.

---

## Part B — Training script: `train_codebert.py`

Create this file in your project (`C:\Users\8000392\source\repos\Python\train_codebert.py`):

```python
import os
import sys
import json
import pandas as pd
from io import StringIO
from azure.storage.blob import BlobServiceClient
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# --------------------------------------------------
# Configuration (from env, injected by the PyTorchJob manifest)
# --------------------------------------------------

BASE_MODEL = os.getenv("BASE_MODEL", "microsoft/codebert-base")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/mnt/output/codebert-new")

BLOB_CONNECTION_STRING = os.environ["BLOB_CONNECTION_STRING"]
BLOB_CONTAINER = os.environ["BLOB_CONTAINER"]
BLOB_NAME = os.environ["BLOB_NAME"]  # e.g. "dataset.csv", columns: code,label

MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "256"))
EPOCHS = int(os.getenv("EPOCHS", "3"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-5"))
VAL_SPLIT = float(os.getenv("VAL_SPLIT", "0.1"))

print("=" * 60)
print("CodeBERT fine-tuning job starting")
print("Base model :", BASE_MODEL)
print("Output dir :", OUTPUT_DIR)
print("Blob       :", BLOB_CONTAINER, "/", BLOB_NAME)
print("Epochs     :", EPOCHS, "Batch size:", BATCH_SIZE, "LR:", LEARNING_RATE)
print("=" * 60)

# --------------------------------------------------
# 1. Download dataset from Azure Blob Storage
# --------------------------------------------------

print("Downloading dataset from Blob Storage...")

blob_service = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
container_client = blob_service.get_container_client(BLOB_CONTAINER)
blob_client = container_client.get_blob_client(BLOB_NAME)

raw_bytes = blob_client.download_blob().readall()

if BLOB_NAME.lower().endswith(".json"):
    df = pd.read_json(StringIO(raw_bytes.decode("utf-8")))
else:
    df = pd.read_csv(StringIO(raw_bytes.decode("utf-8")))

assert "code" in df.columns and "label" in df.columns, \
    "Dataset must have 'code' and 'label' columns"

df = df.dropna(subset=["code", "label"])
df["label"] = df["label"].astype(int)

print(f"Loaded {len(df)} rows. Label distribution:\n{df['label'].value_counts()}")

# --------------------------------------------------
# 2. Train/val split
# --------------------------------------------------

train_df, val_df = train_test_split(
    df, test_size=VAL_SPLIT, random_state=42, stratify=df["label"]
)

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

print(f"Train rows: {len(train_ds)}  Val rows: {len(val_ds)}")

# --------------------------------------------------
# 3. Tokenizer + model (fresh from base, not from codebert-7.0)
# --------------------------------------------------

print("Loading base model:", BASE_MODEL)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

def tokenize_fn(batch):
    return tokenizer(
        batch["code"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
    )

train_ds = train_ds.map(tokenize_fn, batched=True)
val_ds = val_ds.map(tokenize_fn, batched=True)

train_ds = train_ds.rename_column("label", "labels")
val_ds = val_ds.rename_column("label", "labels")

keep_cols = ["input_ids", "attention_mask", "labels"]
train_ds.set_format(type="torch", columns=keep_cols)
val_ds.set_format(type="torch", columns=keep_cols)

# --------------------------------------------------
# 4. Metrics
# --------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
    }

# --------------------------------------------------
# 5. Train
# --------------------------------------------------

training_args = TrainingArguments(
    output_dir="/tmp/checkpoints",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,
    logging_steps=20,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

print("Final evaluation:")
metrics = trainer.evaluate()
print(json.dumps(metrics, indent=2))

# --------------------------------------------------
# 6. Save final model + tokenizer (same shape as codebert-7.0/)
# --------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, "eval_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("Model saved to:", OUTPUT_DIR)
print("Training job complete.")
```

This mirrors the loading/labeling conventions already in `app.py` (2-label
classifier, 0=Human/1=AI, `max_length=256`), so the output checkpoint is a
drop-in replacement for `codebert-7.0/`.

---

## Part C — Training image: `Dockerfile.train`

```dockerfile
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN pip install --no-cache-dir \
    transformers==4.41.2 \
    datasets==2.20.0 \
    accelerate==0.31.0 \
    azure-storage-blob==12.20.0 \
    scikit-learn==1.5.0 \
    pandas==2.2.2

COPY train_codebert.py /app/train_codebert.py

ENTRYPOINT ["python", "/app/train_codebert.py"]
```

Confirm `usernode03`'s NVIDIA driver supports CUDA 12.1 (`nvidia-smi` on the
node, or check `kubectl describe node usernode03` for the driver version
label) before building — adjust the base tag if you need a different CUDA
version.

Build and push (replace `<acr>` with your ACR login server, e.g. `myacr.azurecr.io`):

```bash
docker build -t <acr>/codebert-trainer:v1 -f Dockerfile.train .
docker push <acr>/codebert-trainer:v1
```

---

## Part D — Secret for Blob access

```bash
kubectl create secret generic blob-dataset-secret \
  --namespace ns-models \
  --from-literal=connection-string='<AZURE_STORAGE_CONNECTION_STRING>'
```

---

## Part E — PVC for the trained checkpoint: `codebert-train-pvc.yaml`

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: codebert-output-pvc
  namespace: ns-models
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
  storageClassName: managed-csi  # adjust to your cluster's default/available storage class
```

Check available storage classes first: `kubectl get storageclass`.

---

## Part F — The training job: `codebert-pytorchjob.yaml`

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: codebert-finetune
  namespace: ns-models
spec:
  cleanPodPolicy: None
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: Never
      template:
        spec:
          nodeSelector:
            agentpool: usernode03        # replace with the real label/value from `kubectl get nodes --show-labels`
          containers:
            - name: pytorch
              image: <acr>/codebert-trainer:v1
              env:
                - name: BASE_MODEL
                  value: "microsoft/codebert-base"
                - name: OUTPUT_DIR
                  value: "/mnt/output/codebert-new"
                - name: BLOB_CONTAINER
                  value: "codebert-datasets"
                - name: BLOB_NAME
                  value: "dataset.csv"
                - name: EPOCHS
                  value: "3"
                - name: BATCH_SIZE
                  value: "8"
                - name: LEARNING_RATE
                  value: "2e-5"
                - name: BLOB_CONNECTION_STRING
                  valueFrom:
                    secretKeyRef:
                      name: blob-dataset-secret
                      key: connection-string
              resources:
                limits:
                  nvidia.com/gpu: 1
                requests:
                  nvidia.com/gpu: 1
                  cpu: "2"
                  memory: 8Gi
              volumeMounts:
                - name: output-vol
                  mountPath: /mnt/output
          volumes:
            - name: output-vol
              persistentVolumeClaim:
                claimName: codebert-output-pvc
```

Run it:

```bash
kubectl apply -f codebert-train-pvc.yaml
kubectl apply -f codebert-pytorchjob.yaml

# Watch status
kubectl get pytorchjobs -n ns-models
kubectl get pods -n ns-models -l training.kubeflow.org/job-name=codebert-finetune

# Stream logs (replace pod name from the get pods output above)
kubectl logs -f codebert-finetune-master-0 -n ns-models
```

Expect to see: dataset download confirmation, row counts/label distribution,
per-epoch training loss and eval metrics (accuracy/F1/precision/recall), then
`Model saved to: /mnt/output/codebert-new` and `Training job complete.`

The job finishes when the pod's `PyTorchJob` status shows `Succeeded`:

```bash
kubectl get pytorchjob codebert-finetune -n ns-models -o jsonpath='{.status.conditions[-1].type}'
```

---

## Part G — Serve the new checkpoint with KServe

KServe's `huggingfaceserver` runtime can read model weights directly off a PVC
and supports `sequence_classification` as a task, which matches this
model's architecture (`RobertaForSequenceClassification`, 2 labels) — so no
custom predictor/transformer is needed.

**Important:** PVC volumes are mounted **read-only** by KServe's storage
initializer by default. Since the training job already wrote the checkpoint
to `codebert-output-pvc`, that's fine for serving — just reference the same PVC.

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: codebert-classifier
  namespace: ns-models
spec:
  predictor:
    model:
      modelFormat:
        name: huggingface
      storageUri: "pvc://codebert-output-pvc/codebert-new"
      args:
        - --model_name=codebert
        - --task=sequence_classification
        - --backend=huggingface
      resources:
        requests:
          cpu: "2"
          memory: 4Gi
        limits:
          cpu: "2"
          memory: 4Gi
```

(Add `nvidia.com/gpu: 1` to `resources` if you want GPU-backed inference too —
not required for a 125M-param model at low QPS.)

```bash
kubectl apply -f codebert-inferenceservice.yaml
kubectl get inferenceservice codebert-classifier -n ns-models
```

Wait for `READY` to show `True`, then test:

```bash
curl -X POST http://<isvc-url>/v1/models/codebert:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": ["def add(a, b):\n    return a + b"]}'
```

Expected response shape:

```json
{"predictions": [{"label": "LABEL_1", "score": 0.93}]}
```

Note the generic `huggingfaceserver` runtime returns `LABEL_0`/`LABEL_1`
rather than your custom `HUMAN`/`AI` strings from `app.py:37-40` — either
map that client-side, or keep using your existing `app.py` FastAPI service
(pointed at `OUTPUT_DIR` via `MODEL_PATH` env var) if you need the exact
response shape (`prediction`, `label`, `confidence`, `prob_ai`, chunked
inference) it already provides. Both can read the same checkpoint directory.

---

## Part H — Retrieving the checkpoint without KServe (optional)

If you just want the files copied out of the cluster (e.g. to replace
`codebert-7.0/` locally), spin up a throwaway pod mounting the same PVC:

```bash
kubectl run pvc-access --rm -it --restart=Never -n ns-models \
  --image=busybox --overrides='
{
  "spec": {
    "containers": [{
      "name": "pvc-access",
      "image": "busybox",
      "command": ["sleep", "3600"],
      "volumeMounts": [{"name": "output-vol", "mountPath": "/mnt/output"}]
    }],
    "volumes": [{"name": "output-vol", "persistentVolumeClaim": {"claimName": "codebert-output-pvc"}}]
  }
}'
```

Then, from another terminal:

```bash
kubectl cp ns-models/pvc-access:/mnt/output/codebert-new ./codebert-new-local
```

---

## Part I — CI/CD with Azure DevOps Pipelines

Replaces manually running `docker build` / `docker push` / `kubectl apply`
with a pipeline triggered on every commit. Two variants, depending on whether
Kubeflow Pipelines is installed:

- **I.1** — manifest-based (what's actually running today: `PyTorchJob` +
  `InferenceService`, no KFP installed).
- **I.2** — KFP-based (forward-looking, for the drift-gated retrain pipeline
  discussed earlier, if/when KFP gets installed).

### One-time setup (both variants)

In Azure DevOps → Project Settings → Service connections, create:

1. **`acr-service-connection`** — type "Docker Registry", pointing at your
   ACR. Used to push the training image.
2. **`aks-service-connection`** — type "Kubernetes Service Connection",
   pointing at the `LMAPAZ1AKSNPDAI` cluster. This is also how the
   manifest-based pipeline avoids the local Netskope TLS problem you hit
   earlier — the Microsoft-hosted agent reaches the AKS API server directly,
   not through your intercepted corporate network.

---

### I.1 — Manifest-based CI/CD (current setup, no KFP)

#### Templated manifests

Duplicate `codebert-pytorchjob.yaml` and `codebert-inferenceservice.yaml`
(from Parts F and G) as `.tmpl.yaml` files with placeholders, so every run
gets a unique image tag and job name instead of colliding with the previous
run's `PyTorchJob` object:

**`codebert-pytorchjob.tmpl.yaml`** — same as Part F's manifest, except:
```yaml
metadata:
  name: __JOB_NAME__
  namespace: ns-models
spec:
  ...
          containers:
            - name: pytorch
              image: __IMAGE__
```

**`codebert-inferenceservice.tmpl.yaml`** — same as Part G's manifest, except:
```yaml
      storageUri: "pvc://codebert-output-pvc/__MODEL_SUBDIR__"
```

#### `azure-pipelines.yml`

```yaml
trigger:
  branches:
    include:
      - main
  paths:
    include:
      - train_codebert.py
      - Dockerfile.train
      - codebert-pytorchjob.tmpl.yaml
      - codebert-inferenceservice.tmpl.yaml

variables:
  acrName: '<acr-login-server>'              # e.g. myacr.azurecr.io
  imageRepo: 'codebert-trainer'
  jobName: 'codebert-finetune-$(Build.BuildId)'
  modelSubdir: 'codebert-$(Build.BuildId)'

stages:
  # ---------------------------------------------------------------
  - stage: Build
    jobs:
      - job: BuildAndPush
        pool:
          vmImage: ubuntu-latest
        steps:
          - task: Docker@2
            inputs:
              containerRegistry: 'acr-service-connection'
              repository: '$(imageRepo)'
              command: 'buildAndPush'
              Dockerfile: 'Dockerfile.train'
              tags: |
                $(Build.BuildId)

  # ---------------------------------------------------------------
  - stage: Train
    dependsOn: Build
    jobs:
      - job: SubmitTrainingJob
        pool:
          vmImage: ubuntu-latest
        steps:
          - script: |
              sed -e "s|__IMAGE__|$(acrName)/$(imageRepo):$(Build.BuildId)|g" \
                  -e "s|__JOB_NAME__|$(jobName)|g" \
                  codebert-pytorchjob.tmpl.yaml > codebert-pytorchjob.rendered.yaml
            displayName: 'Render PyTorchJob manifest'

          - task: KubernetesManifest@1
            displayName: 'Apply PVC + PyTorchJob'
            inputs:
              action: 'deploy'
              connectionType: 'kubernetesServiceConnection'
              kubernetesServiceConnection: 'aks-service-connection'
              namespace: 'ns-models'
              manifests: |
                codebert-train-pvc.yaml
                codebert-pytorchjob.rendered.yaml

          - task: Kubernetes@1
            displayName: 'Wait for PyTorchJob to complete'
            inputs:
              connectionType: 'Kubernetes Service Connection'
              kubernetesServiceEndpoint: 'aks-service-connection'
              namespace: 'ns-models'
              command: 'wait'
              arguments: '--for=condition=Succeeded pytorchjob/$(jobName) --timeout=2h'

  # ---------------------------------------------------------------
  - stage: Deploy
    dependsOn: Train
    condition: succeeded()
    jobs:
      - job: DeployToKServe
        pool:
          vmImage: ubuntu-latest
        steps:
          - script: |
              sed -e "s|__MODEL_SUBDIR__|$(modelSubdir)|g" \
                  codebert-inferenceservice.tmpl.yaml > codebert-inferenceservice.rendered.yaml
            displayName: 'Render InferenceService manifest'

          - task: KubernetesManifest@1
            displayName: 'Apply InferenceService'
            inputs:
              action: 'deploy'
              connectionType: 'kubernetesServiceConnection'
              kubernetesServiceConnection: 'aks-service-connection'
              namespace: 'ns-models'
              manifests: |
                codebert-inferenceservice.rendered.yaml
```

Notes:
- `kubectl wait --for=condition=Succeeded` assumes the Training Operator
  version installed (Part A, `v1.8.1`) reports a `Succeeded` condition type on
  `PyTorchJob` status — confirm with `kubectl get pytorchjob <name> -n
  ns-models -o yaml` after one manual run; older versions may use a different
  condition name.
- The `Deploy` stage only runs if `Train` succeeded (`condition:
  succeeded()`), so a failed or still-training job never gets served.
- Each run's `PyTorchJob` and PVC subdirectory are uniquely named via
  `$(Build.BuildId)`, so reruns don't collide with prior training runs; old
  `PyTorchJob` objects can be cleaned up separately (e.g. a scheduled
  `kubectl delete pytorchjob --field-selector status.phase=Succeeded`) since
  they aren't auto-deleted (`cleanPodPolicy: None`).

---

### I.2 — KFP-based CI/CD (forward-looking, if KFP gets installed)

This path replaces `kubectl apply` with **compiling** the Python pipeline
definition to IR YAML and submitting it to the KFP backend's API, rather than
applying raw K8s manifests directly.

#### `compile_pipeline.py`

```python
from kfp import dsl, kubernetes
from kfp.compiler import Compiler


@dsl.component(base_image="<acr-login-server>/codebert-trainer:latest")
def check_drift(threshold: float) -> float:
    # Compute drift score against recent production predictions/inputs.
    # Placeholder — replace with your real drift computation.
    import random
    return random.uniform(0.0, 0.5)


@dsl.component(base_image="<acr-login-server>/codebert-trainer:latest")
def train_codebert(output_dir: str):
    import subprocess
    subprocess.run(["python", "/app/train_codebert.py"], check=True, env={
        "OUTPUT_DIR": output_dir,
    })


@dsl.component(base_image="<acr-login-server>/codebert-trainer:latest")
def evaluate_and_register(model_dir: str) -> float:
    import json
    with open(f"{model_dir}/eval_metrics.json") as f:
        metrics = json.load(f)
    return metrics["eval_f1"]


@dsl.component(base_image="<acr-login-server>/codebert-trainer:latest")
def deploy_to_kserve(model_dir: str):
    # Patch the InferenceService storageUri to point at the new checkpoint,
    # e.g. via the kubernetes python client or `kubectl patch`.
    print(f"Deploying model from {model_dir} to KServe")


@dsl.pipeline(name="codebert-drift-retrain")
def pipeline(drift_threshold: float = 0.3, f1_threshold: float = 0.85):
    drift_task = check_drift(threshold=drift_threshold)

    with dsl.If(drift_task.output > drift_threshold):
        train_task = train_codebert(output_dir="/mnt/output/codebert-new")
        train_task.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1)
        kubernetes.add_node_selector(train_task, "agentpool", "usernode03")

        eval_task = evaluate_and_register(model_dir=train_task.output)

        with dsl.If(eval_task.output > f1_threshold):
            deploy_to_kserve(model_dir=train_task.output)


if __name__ == "__main__":
    Compiler().compile(pipeline, "pipeline.yaml")
```

#### `azure-pipelines.yml`

```yaml
trigger:
  branches:
    include:
      - main
  paths:
    include:
      - compile_pipeline.py
      - train_codebert.py
      - Dockerfile.train

variables:
  acrName: '<acr-login-server>'

stages:
  # ---------------------------------------------------------------
  - stage: Build
    jobs:
      - job: BuildAndPush
        pool:
          vmImage: ubuntu-latest
        steps:
          - task: Docker@2
            inputs:
              containerRegistry: 'acr-service-connection'
              repository: 'codebert-trainer'
              command: 'buildAndPush'
              Dockerfile: 'Dockerfile.train'
              tags: |
                $(Build.BuildId)
                latest

  # ---------------------------------------------------------------
  - stage: KFPPipeline
    dependsOn: Build
    jobs:
      - job: CompileAndRunKFP
        pool:
          vmImage: ubuntu-latest
        variables:
          - group: kfp-secrets   # holds KFP_ENDPOINT, KFP_AUTH_TOKEN (Key Vault-backed)
        steps:
          - script: pip install kfp==2.*
            displayName: 'Install KFP SDK'

          - script: python compile_pipeline.py
            displayName: 'Compile pipeline to IR YAML'

          - script: |
              python - <<EOF
              from kfp.client import Client
              client = Client(host="$(KFP_ENDPOINT)", existing_token="$(KFP_AUTH_TOKEN)")
              client.create_run_from_pipeline_package(
                  "pipeline.yaml",
                  arguments={"drift_threshold": 0.3, "f1_threshold": 0.85},
                  experiment_name="codebert-finetune",
              )
              EOF
            displayName: 'Upload and trigger KFP run'
```

Notes:
- **Auth is different from I.1**: the KFP client talks to the KFP API
  server's HTTP endpoint (behind the cluster's ingress), not the K8s API
  server — so this stage uses `KFP_ENDPOINT`/`KFP_AUTH_TOKEN` as secret
  pipeline variables (a variable group backed by Azure Key Vault), not the
  `aks-service-connection` used in I.1.
- **Recurring runs are registered once, not per-CI-run.** The drift-check
  cadence (e.g. daily) is set up with `client.create_recurring_run(...)`
  pointing at an uploaded pipeline version — do this as a one-time
  (or pipeline-definition-change-triggered) step, separate from the
  `create_run_from_pipeline_package` call above, which triggers one
  immediate run on every commit. Conflating the two means either re-creating
  a duplicate recurring schedule on every commit, or never actually
  scheduling one.
- `train_codebert`'s container command would need adjusting from the
  standalone `train_codebert.py` (Part B) to accept `output_dir` as an
  argument rather than purely via env var, if you want the KFP-tracked
  `model_dir` output to flow correctly between components.

---

## Verification checklist

- [ ] `kubectl get pods -n kubeflow` → training-operator pod `Running`
- [ ] `kubectl get crd | grep pytorchjobs` → CRD present
- [ ] `kubectl get pytorchjob codebert-finetune -n ns-models` → `Succeeded`
- [ ] Training logs show decreasing loss and final F1/accuracy printed
- [ ] `codebert-new/` directory in the PVC contains `config.json`,
      `model.safetensors`, tokenizer files (same shape as `codebert-7.0/`)
- [ ] `kubectl get inferenceservice codebert-classifier -n ns-models` → `READY: True`
- [ ] `curl .../v1/models/codebert:predict` returns a sane label/score on a
      known human-written and a known AI-generated snippet before you retire
      the current `app.py`-based deployment
