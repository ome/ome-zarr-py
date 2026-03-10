# Zarr Concepts

[Zarr](https://zarr.dev/) is the underlying storage format for OME-NGFF. Understanding
Zarr concepts helps when working with OME-Zarr data.

## Key Concepts

### Stores
Zarr arrays can be stored in various backends:
- **Directory store**: Files on local disk (most common)
- **S3 store**: Cloud object storage (Amazon S3, MinIO, etc.)
- **HTTP store**: Read-only access via HTTP/HTTPS
- **Memory store**: In-memory storage for testing

### Groups and Arrays
Zarr organizes data hierarchically:
- **Groups**: Containers that can hold arrays and other groups (like folders)
- **Arrays**: N-dimensional data chunks with metadata

### Chunks
Large arrays are divided into chunks for efficient access:
- Each chunk is stored as a separate file/object
- Only needed chunks are loaded into memory
- Chunk shape affects performance for different access patterns

```python
import zarr

# Create array with specific chunk shape
arr = zarr.zeros((10000, 10000), chunks=(1000, 1000))
```

### Zarr v2 vs v3

OME-NGFF v0.4 uses Zarr v2, while OME-NGFF v0.5 uses Zarr v3:

| Feature | Zarr v2 | Zarr v3 |
|---------|---------|---------|
| Metadata file | `.zarray`, `.zgroup` | `zarr.json` |
| Sharding | No | Yes |
| Codecs | Limited | Extensible |

## Resources

- [Zarr Tutorial](https://zarr.readthedocs.io/en/stable/tutorial.html)
- [Zarr v3 Spec](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html)
- [zarr-python Documentation](https://zarr.readthedocs.io/)
