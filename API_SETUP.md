# API and service setup

## What is already local

The detection, report, analytics, and CSV endpoints are served by this FastAPI app. They do not need third-party API keys. The detector uses `models/best.pt`; report data is stored in `roadsentinel.db` (SQLite). Start the backend with:

```powershell
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://localhost:8000` for the frontend and `http://localhost:8000/api/docs` for the API. The frontend currently uses `const API = ''`, which means same-origin requests. If you host the frontend separately, set `API` to the backend origin and configure FastAPI CORS for that exact frontend origin.

## External services used by the frontend

| Service | Current use | Setup needed |
| --- | --- | --- |
| Google Identity Services | Google sign-in UI is present, but the button is intentionally marked unconfigured. | Create a Google Cloud project and Web OAuth client ID, add the deployed origins, initialize GIS with the client ID, send the returned ID credential to `/auth/google`, and verify its signature, audience, issuer, and expiry on the server before creating a session. See [Google's client ID setup](https://developers.google.com/identity/gsi/web/guides/get-google-api-clientid). |
| Map tiles (CARTO) | Leaflet renders the report map using CARTO tiles. | Choose a tile provider and follow its current access and attribution terms. Keep attribution visible on the map. CARTO's [basemap terms](https://carto.com/legal/basemap-terms/) and [attribution guidance](https://carto.com/attribution/) describe its requirements. |
| Nominatim / OpenStreetMap | Reverse geocoding turns the user's GPS position into an address. | No key is currently configured. The public Nominatim endpoint has a usage policy and is not intended as an unrestricted production geocoding backend; review the [Nominatim policy](https://operations.osmfoundation.org/policies/nominatim/) and use a hosted or self-managed service if your traffic needs it. |
| CDN assets | Tailwind, Leaflet, and Google Fonts load from third-party CDNs. | No API keys; ensure your deployment can reach these hosts, or bundle the assets locally for production. |

## Authentication status

Local signup now hashes passwords with Argon2 and login verifies the hash. Signup requires a name and an eight-character minimum password. Google sign-in is disabled in the UI and `/auth/google` returns `501` until real Google ID token verification is implemented. The app currently returns no session token, and its report/analytics endpoints do not require authentication; treat this as a local development flow, not production access control. Before deployment, add signed sessions, protect private endpoints, and rate-limit authentication. Keep all service secrets on the backend, never in `app/static/index.html`.

## Other integrations not present yet

- Email/SMS notifications: the UI currently has no notification delivery API. Choose a provider, store its secret on the backend, and add a backend notification endpoint or background job.
- Municipal dispatch: report status updates are local database changes through `PATCH /reports/{id}`. No city or municipal system is connected; obtain its API documentation and credentials from the municipality before implementing a connector.
- Production database: SQLite is configured for local use. For a deployed multi-user service, configure PostgreSQL (and PostGIS if spatial queries are needed) with a server-side `DATABASE_URL` environment variable.
