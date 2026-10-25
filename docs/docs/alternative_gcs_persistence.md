# Alternative: Google Cloud Storage + Cloud Storage FUSE

The canonical reference implementation stores LangGraph checkpoints in Firestore. Assignment 2 also accepts durable persistence backed by **Google Cloud Storage (GCS)** mounted into Cloud Run through native **Cloud Storage FUSE/GCSFuse**.

## Architecture

```text
Cloud Run service/instance
        |
        v
mounted path (for example /mnt/state)
        |
        v
GCS bucket
```

## Configuration surface

`env.example` includes:

```text
GCS_STATE_BUCKET=your-state-bucket
GCS_STATE_MOUNT=/mnt/state
```

Students choosing this path should configure the Cloud Run volume mount using the current Google Cloud CLI or Console instructions, grant the service identity access to the bucket, and make the application actually read/write the state it claims to persist through the mounted path.

The grading requirement is behavioral: if the project claims durable multi-turn state, that state must survive Cloud Run instance replacement. GCS/GCSFuse is one accepted way to provide that durability.

Cloud Run Instances may also be used; the storage requirement is satisfied by the mounted GCS bucket, not by process memory.
