
---

# 🛡️ **2️⃣ SECURITY.md (for sensitive info)**

Create a separate file `SECURITY.md` in your root folder:

```markdown
# 🔒 Security & Commit Guidelines

To ensure responsible data handling and repository hygiene, follow these guidelines:

---

## 🚫 DO NOT COMMIT

- `.env` files (containing credentials or API keys)
- API tokens, secret keys, passwords
- Raw or private datasets (use processed or anonymized data)
- Machine-generated large binary files (`.pdf`, `.laz`, `.zip`)
- `/data/raw/` contents
- `/state/` contents (hash tracking files)
- Virtual environments (`.venv/`, `__pycache__/`)
- Model weights not intended for open distribution

---

## ✅ SAFE TO COMMIT

- Source code (`.py`)
- Configuration templates (`sources.yaml`)
- Markdown files (`.md`)
- Scripts, utilities, and notebooks for reproducible results
- Processed datasets that contain only public information

---

## 🧠 Reporting Issues

If you find a security vulnerability (like exposed credentials), **do not open a public GitHub issue**.  
Instead, email the project maintainer at:
> 📧 ramakrishnah1@montclair.edu

---

## 🛡️ Policy

All scraping, downloading, or automated requests respect the target website’s `robots.txt`, rate limits, and academic fair-use guidelines.
