# Cloudinary storage

Cloudinary storage is optional. Without credentials, the application retains
its existing local-file behavior. With credentials, every successfully
evaluated scan uploads four artifacts:

- original uploaded OMR image
- geometrically corrected OMR image
- evaluated/bubble-debug OMR image
- evaluation result JSON

## Environment variables

Use either a single Cloudinary URL:

```text
CLOUDINARY_URL=cloudinary://API_KEY:API_SECRET@CLOUD_NAME
```

or the three individual variables:

```text
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
```

The optional `CLOUDINARY_OMR_FOLDER` changes the root asset folder. It defaults
to `omr-scanner`.

Never commit the API secret or expose it in browser JavaScript. Set these
variables in the production hosting environment. OMR images may contain
student personal information, so configure Cloudinary access controls to suit
your privacy requirements.
