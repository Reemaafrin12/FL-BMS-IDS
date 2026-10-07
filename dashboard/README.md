# EVSE IDS demo dashboard

This minimal dashboard has three services. Start them in this order:

1. **ML service** (from `dashboard/ml-service`): install `requirements.txt`, ensure `../../model.pkl` exists, then run `uvicorn app:app --reload --port 8000`.
2. **API server** (from `dashboard/server`): copy `.env.example` to `.env`, set a MongoDB URI, run `npm install`, then run `npm run dev`.
3. **React client** (from `dashboard/client`): run `npm install`, then `npm run dev`. Vite proxies `/api` to port 5000.

Create the inference model from the project root before starting the ML service:

```powershell
python train_baseline.py --save-model model.pkl
```

With all services running, replay real dataset rows as a demo feed:

```powershell
python dashboard/simulate_feed.py
```

The feed reads `data/clients/client_3.csv` one row at a time and sends one
reading every two seconds. It is a simulated replay of real dataset rows, **not**
a connection to a live physical EVSE sensor.
