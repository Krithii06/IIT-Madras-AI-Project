# Deployment

> **What actually shipped:** everything on Vercel, in one project, at
> <https://plant-disease-classification-rosy.vercel.app>. Static build plus Python
> serverless functions in `frontend/api/`, same origin, no CORS. Nothing below is
> needed for that — Vercel deploys it from a push to `main`. See the Deployment
> section of the top-level README for the three non-obvious things about that build.
>
> This document describes the **original two-host design**, kept because the
> Dockerfile and `render.yaml` are still in the repository and still work. It is the
> route to take if the API needs a long-running container rather than a function —
> for example to hold state, or to run a model too large for a 250 MB bundle.

Two pieces go to two different hosts: the FastAPI inference service to Render, and
the built React bundle to Vercel. Both have a free tier that does not require a card.

Deploy the backend first — the frontend build needs the backend URL, and the
backend's CORS setting needs the frontend URL, so there is one crossing-over step
at the end.

## Why these two

Hugging Face Spaces was the original target and the Dockerfile still works there,
but as of 2026 the Hub documentation states that *"Gradio and Docker Spaces run on
compute and require a paid plan to create: PRO for personal accounts"*. Only Static
Spaces remain free, and a static Space cannot run the model. Render still offers a
free Docker web service, so that is what these instructions use.

Check the current terms before relying on any of this — free tiers move.

### What the free tier actually gives you

| | Render free web service | Vercel Hobby |
|---|---|---|
| Cost | $0 | $0 |
| RAM / CPU | 512 MB / 0.1 CPU | static hosting |
| Sleeps when idle | yes, after 15 min | no |
| Cold start | ~60 s | n/a |
| Monthly allowance | 750 instance hours | generous for static |

The 512 MB limit is the reason the API serves ONNX rather than torch, and the
cold start is the reason the frontend shows a "waking the prediction service"
notice and uses a 90-second request timeout.

## 1. Push the repository to GitHub

Both hosts deploy from a Git repository.

```bash
git remote add origin https://github.com/<you>/plant-disease-classification.git
git push -u origin main
```

Confirm that `models/export/` is present in the pushed tree. It is the only model
directory that is committed, and the Docker build copies it into the image — if it
is missing the container starts but `/health` reports `degraded`.

## 2. Backend on Render

1. Sign in at <https://render.com> with the GitHub account holding the repo.
2. **New → Web Service**, then select the repository.
3. Render reads `render.yaml` and should fill these in. Confirm them:
   - Language / runtime: **Docker**
   - Dockerfile path: `./backend/Dockerfile`
   - Docker build context: `.` (repository root, **not** `backend/`)
   - Health check path: `/health`
   - Instance type: **Free**
4. Leave `ALLOWED_ORIGINS` unset for now. Create the service.
5. Wait for the first build. It takes a few minutes, mostly installing
   onnxruntime.
6. Note the service URL, of the form `https://plant-disease-api.onrender.com`.

Check it before moving on:

```bash
curl https://<your-service>.onrender.com/health
```

Expect `{"status":"ok","model_loaded":true}`. If it says `degraded`, the image
built but `models/export/` did not make it in — see step 1.

The interactive API docs are at `https://<your-service>.onrender.com/docs`.

## 3. Frontend on Vercel

1. Sign in at <https://vercel.com> with the same GitHub account.
2. **Add New → Project**, import the repository.
3. Set:
   - Root directory: `frontend`
   - Framework preset: **Vite** (auto-detected)
   - Build command: `npm run build`
   - Output directory: `dist`
4. Add an environment variable:
   - `VITE_API_URL` = `https://<your-service>.onrender.com` (no trailing slash)
5. Deploy, then note the resulting `https://<project>.vercel.app` URL.

`VITE_API_URL` is read at **build** time, not at runtime. If you change it later
you have to redeploy for it to take effect.

## 4. Let the frontend talk to the backend

The API only accepts browser requests from origins it has been told about, so the
last step is to hand it the Vercel URL.

1. In Render, open the service → **Environment**.
2. Set `ALLOWED_ORIGINS` to the exact Vercel origin, scheme included and no
   trailing slash:

   ```
   https://<project>.vercel.app
   ```

   Several origins can be given as a comma-separated list, which is useful if you
   also want to allow Vercel preview deployments.
3. Save. Render restarts the service.

Then open the Vercel URL, upload a leaf image and confirm a prediction comes back.
The first request after an idle period will take about a minute while the backend
wakes; the UI says so while it waits.

## Troubleshooting

**The page loads but predicting fails with a network error.** Almost always CORS.
Open the browser console — a blocked request names the origin it tried. Make sure
that exact string is in `ALLOWED_ORIGINS`, without a trailing slash.

**`/health` returns `degraded`.** The model bundle is missing. Confirm
`models/export/model.onnx`, `class_mapping.json` and `preprocess.json` are all
committed, and that the Docker build context is the repository root.

**First request times out.** A cold start plus a large upload can exceed the
90-second client timeout. Try again once the service is awake.

**Build runs out of memory.** The free instance builds with limited RAM. Nothing
in this image should come close, but if it happens, confirm you are not
accidentally installing `torch` — only `backend/requirements.txt` should be used.

## Keeping it awake

A free service sleeping after 15 minutes is a poor demo experience. An uptime
pinger hitting `/health` every 10 minutes will keep it warm, but it also burns the
750 monthly instance hours — roughly 31 days of continuous uptime, so a single
always-on service just about fits and leaves nothing spare. Decide which you would
rather have before setting one up.
