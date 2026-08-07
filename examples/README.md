# Sifter Examples

This directory contains example container definitions and a sample manifest to help you get started with Sifter.

## Directory Structure

```
examples/
├── sifter.yaml              # Example manifest defining containers
├── definitions/
│   ├── base/                # Base image (uses build args)
│   │   └── base.def         # Apptainer definition file
│   ├── app/                 # Application layer (uses build args)
│   │   └── app.def          # Definition with {{ BASE_IMAGE }} placeholder
│   └── adhoc/               # Ad-hoc build examples (no build args)
│       ├── base.def         # Simple base image
│       └── app.def          # App layer using {{ BASE_IMAGE }}
└── README.md                # This file
```

## Quick Start

### Option A: Manifest-Based Builds (Recommended)

Use `sifter.yaml` to define builds with version tracking and build arguments.
You address a build by its **name** — the part before the last `_` (e.g.
`base-hello`), not the full `name_version` tag.

```bash
# Dry run - see what would be built
sifter build base-hello --dry-run

# Build a base image
sifter build base-hello

# Build an app (automatically builds dependencies first)
sifter build app-hello

# Build all containers (the whole matrix, as one dependency chain)
sifter build --all
```

### Option B: Ad-hoc Builds (No Manifest)

Build directly from definition files without a manifest. Useful for quick tests.

> **Note**: Ad-hoc builds do not support build arguments. Use hardcoded values in definition files,
> or use Option A for builds requiring arguments.

```bash
# Single definition
sifter build definitions/adhoc/base.def --tag my-base

# Multiple steps (chained - second uses first as base image)
sifter build --steps definitions/adhoc/base.def,definitions/adhoc/app.def --tag my-app
```

### Check Build Status

```bash
# See running and recent jobs
sifter status

# View logs for a specific job
sifter logs <job_id>
```

### Publish to a Registry (Optional)

With a registry configured (a signed OCI/ECR registry is recommended — see the
[Configuration section of the top-level README](../README.md#configuration)):

```bash
# Push a specific image to the remote registry (signed)
sifter push app-hello_0.0.1.sif

# Push every manifest image present locally
sifter push --all

# Pull an image back (verified); the .sif extension is optional
sifter pull app-hello_0.0.1.sif
```

## Creating Your Own Containers

### 1. Create a Definition Directory

```bash
mkdir -p definitions/mycontainer
```

### 2. Create the Definition File (`definitions/mycontainer/mycontainer.def`)

```singularity
Bootstrap: docker
# Pin the base by digest for reproducible, supply-chain-safe builds.
From: ubuntu:24.04@sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90

%arguments
    MY_ARG=default_value

%post
    echo "Building with MY_ARG={{ MY_ARG }}"
    # Install your dependencies here

%runscript
    exec "$@"
```

### 3. Add to `sifter.yaml`

```yaml
builds:
  # Simple build with no dependencies
  mycontainer-v1_0.0.1:
    steps:
      - path: definitions/mycontainer/mycontainer.def
        args:
          MY_ARG: "value1"

  # Another variant with different args
  mycontainer-v2_0.0.1:
    steps:
      - path: definitions/mycontainer/mycontainer.def
        args:
          MY_ARG: "value2"
```

## Building on Dependencies

To build a container on top of another, use the `{{ BASE_IMAGE }}` placeholder:

```singularity
Bootstrap: localimage
From: {{ BASE_IMAGE }}

%post
    # Your additional setup here
```

And declare the dependency using `base:` in `sifter.yaml`:

```yaml
builds:
  # Base image (no base)
  base-hello_0.0.1:
    steps:
      - path: definitions/base/base.def
        args:
          BASE_MESSAGE: "Hello from base"

  # App that depends on base
  app-hello_0.0.1:
    base: base-hello_0.0.1
    steps:
      - path: definitions/app/app.def
        args:
          APP_MESSAGE: "Hello from app"
```

Sifter will:
1. Build dependencies first (or reuse an existing local build with matching content)
2. Set `BASE_IMAGE` for the dependent definition file
3. Chain SLURM jobs with proper dependencies (`afterok`)
