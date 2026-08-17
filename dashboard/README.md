# Real-Time Monitoring Dashboard

This directory contains a web dashboard for monitoring the arbitrage bot in real time.

## Architecture

- **Backend:** a Node.js server that subscribes to the bot's Redis channel (`bot_dashboard_channel`) and relays messages to the frontend over WebSocket (`socket.io`).
- **Frontend:** a React application (TypeScript) that receives live data over WebSocket and renders it with Chart.js and an activity log component.

## Running It

The dashboard needs both the backend server and the frontend dev server running. Use two separate terminals.

### 1. Start the backend server

In your first terminal:

```bash
cd dashboard/backend
node server.js
```

You should see output similar to:

```
Connected to Redis successfully.
Subscribed to Redis channel: bot_dashboard_channel
Dashboard backend server listening on *:3001
```

This requires a Redis instance reachable at `redis://localhost:6379`.

### 2. Start the frontend dev server

In your second terminal:

```bash
cd dashboard/frontend
npm start
```

This opens a new browser tab at [http://localhost:3000](http://localhost:3000).

### 3. Start the Python bot

In a third terminal, run the bot — paper mode is a good starting point:

```bash
# activate your virtual environment first
source .venv/bin/activate

python -m src.cli.main paper
```

Dashboard publishing is controlled by the `dashboard` block in `config/default.yaml` and must be enabled for data to flow.

## Usage

- Once all three components are running, data appears on the dashboard at [http://localhost:3000](http://localhost:3000).
- The line chart updates whenever the bot detects an arbitrage opportunity.
- The activity log shows every message the bot publishes.
