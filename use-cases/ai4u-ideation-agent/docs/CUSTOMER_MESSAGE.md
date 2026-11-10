**AI4U Ideation Agent — Deployment**

Repository: `<repo-link>`

**Prerequisites** (BTP Cockpit → Entitlements & Instances):
- Entitlements: AI Core `extended`, HANA Cloud `hana`, Object Store `s3-standard`, XSUAA `application`, CF Runtime ≥ 3 GB
- AI Core instance with **GPT-5.4** + **Gemini 2.5 Flash** deployed — [tutorial](https://developers.sap.com/tutorials/ai-core-generative-ai)
- HANA Cloud instance running
- [CF CLI](https://docs.cloudfoundry.org/cf-cli/install-go-cli.html) installed

**Deploy:**
```
cf login -a https://api.cf.<your-region>.hana.ondemand.com --sso
./setup.sh
```

Full guide: `docs/DEPLOYMENT.md`
