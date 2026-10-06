# Deploying to Streamlit Community Cloud

The repo is already deployment-ready (see README "Day 8" for what that
involved) and has a local git history. What's left requires your own
GitHub and Streamlit accounts, so it has to happen on your machine /
browser, not here. Five steps, ~5 minutes:

## 1. Create an empty GitHub repository

Go to https://github.com/new, name it (e.g. `smart-irrigation-system`),
**do not** check "Add a README" (you already have one) — create it empty.

## 2. Push this project to it

From inside the `smart-irrigation-project/` folder (it's already a git repo
with one commit):

```bash
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git branch -M main
git push -u origin main
```

## 3. Sign up / log in at Streamlit Community Cloud

Go to https://share.streamlit.io and sign in with your GitHub account
(free tier is sufficient for this project).

## 4. Deploy

Click **"New app"**, then:
- **Repository:** `<your-username>/<your-repo-name>`
- **Branch:** `main`
- **Main file path:** `dashboard/app.py`

Click **Deploy**. First build takes 2-3 minutes.

## 5. Verify it actually works

Once live, check:
- [ ] The app loads without a "Oh no" error screen (if it does, click
      "Manage app" → logs to see why — nearly always a missing/mismatched
      package, which `requirements.txt` should already prevent per Day 8)
- [ ] Switch the model dropdown to **Neural Network** and get a
      recommendation — confirms the NumPy inference path works in the
      cloud environment, not just this sandbox
- [ ] Toggle **"Use live weather API"** off and on — confirms Open-Meteo is
      actually reachable from Streamlit Cloud's network (it could not be
      tested from this sandbox; see README Day 5)
- [ ] Try a drought scenario (moisture ~15%, manual weather) and confirm
      the safety-override banner appears (Day 6/7's key safety feature)

## Why this should fit the free tier

Day 8 specifically measured and fixed memory usage for this: the deployed
app now peaks at **~290MB RAM** (measured in a clean virtual environment
mirroring what Streamlit Cloud installs), comfortably under the 1GB
free-tier limit. Before the Day 8 fix, TensorFlow alone consumed ~700MB
before the app did anything — this would likely have crashed on deploy.

## If it still fails to deploy

Most likely causes, in order of likelihood:
1. **Main file path typo** — must be exactly `dashboard/app.py`
2. **Open-Meteo blocked from Streamlit Cloud's network** — unlikely (it's a
   public, unauthenticated API) but if the weather panel shows the offline
   fallback warning even with real internet, that's the first thing to
   check in the logs
3. **A dependency version Streamlit Cloud can't resolve** — check the build
   log; `requirements.txt` is pinned to exact versions verified in a clean
   venv (see README Day 8), so this should be unlikely, but package
   availability can change over time

Once deployed, the app gets a public `*.streamlit.app` URL you can put
directly in your final report/presentation and share with evaluators.
