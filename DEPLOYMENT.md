# Deployment Guide: Auto-sync GitHub to Hugging Face Spaces

This guide explains how to set up automatic deployment from GitHub to Hugging Face Spaces.

## Prerequisites

1. A GitHub repository with your Streamlit app
2. A Hugging Face account ([sign up here](https://huggingface.co/join))
3. Git installed locally

## Step 1: Create a Hugging Face Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Configure your Space:
   - **Owner**: Your username or organization
   - **Space name**: Choose a name (e.g., `sams-clustering-demo`)
   - **License**: MIT (or your preferred license)
   - **SDK**: Select **Streamlit**
   - **Hardware**: CPU (free tier)
   - **Visibility**: Public or Private
4. Click **"Create Space"**

## Step 2: Get Your Hugging Face Token

1. Go to [Hugging Face Settings → Access Tokens](https://huggingface.co/settings/tokens)
2. Click **"New token"**
3. Configure the token:
   - **Name**: `github-actions` (or any descriptive name)
   - **Role**: Select **Write** (required for pushing to Spaces)
4. Click **"Generate token"**
5. **Copy the token** (you won't be able to see it again!)

## Step 3: Add Token to GitHub Secrets

1. Go to your GitHub repository
2. Navigate to **Settings → Secrets and variables → Actions**
3. Click **"New repository secret"**
4. Add the secret:
   - **Name**: `HF_TOKEN`
   - **Secret**: Paste your Hugging Face token
5. Click **"Add secret"**

## Step 4: Configure the GitHub Actions Workflow

1. Edit the file `.github/workflows/sync-to-huggingface.yml`
2. Replace the placeholders:
   - `HF_USERNAME`: Your Hugging Face username
   - `SPACE_NAME`: Your Space name (from Step 1)

Example:
```yaml
git remote add hf https://HF_USERNAME:$HF_TOKEN@huggingface.co/spaces/HF_USERNAME/SPACE_NAME
```

Should become:
```yaml
git remote add hf https://myusername:$HF_TOKEN@huggingface.co/spaces/myusername/sams-clustering-demo
```

## Step 5: Prepare Your Repository

Make sure your repository has:

1. **README_HF.md** - This will become README.md on Hugging Face (with metadata header)
2. **app.py** - Your Streamlit application
3. **requirements.txt** - Python dependencies
4. **packages.txt** - System packages (if needed)

The workflow will automatically sync these files to your Space.

## Step 6: Push to GitHub

1. Commit all changes:
   ```bash
   git add .
   git commit -m "Set up Hugging Face auto-deployment"
   ```

2. Push to the main branch:
   ```bash
   git push origin main
   ```

3. The GitHub Action will automatically:
   - Trigger on push to main
   - Sync your repository to Hugging Face Spaces
   - Your app will rebuild automatically on Hugging Face

## Step 7: Verify Deployment

1. Go to your Hugging Face Space: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`
2. Check the **"Building"** status
3. Once complete, your app will be live!

## Alternative: Manual Initial Setup

If you prefer to set up the Space manually first:

1. Clone your Space locally:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   cd YOUR_SPACE_NAME
   ```

2. Copy your app files:
   ```bash
   cp /path/to/app.py .
   cp /path/to/requirements.txt .
   cp /path/to/README_HF.md README.md
   ```

3. Push to Hugging Face:
   ```bash
   git add .
   git commit -m "Initial commit"
   git push
   ```

4. Then enable auto-sync using the GitHub Actions workflow

## Troubleshooting

### Action fails with authentication error
- Verify your `HF_TOKEN` is correct in GitHub Secrets
- Ensure the token has **Write** access
- Check that username and space name are correct in the workflow

### App doesn't start on Hugging Face
- Check the build logs in the Space
- Verify `requirements.txt` includes all dependencies
- Ensure `app_file: app.py` is correct in README_HF.md metadata

### Changes not reflecting
- Check the GitHub Actions tab for workflow status
- Verify the workflow completed successfully
- Hugging Face may take a few minutes to rebuild

## Next Steps

- Customize README_HF.md with your app description
- Add a banner image to your Space
- Configure Space settings (hardware, secrets, etc.)
- Share your Space with the community!

## Files Created

- `.github/workflows/sync-to-huggingface.yml` - GitHub Actions workflow
- `README_HF.md` - Hugging Face Space metadata and description
- `packages.txt` - System dependencies (currently empty)
- `DEPLOYMENT.md` - This deployment guide
