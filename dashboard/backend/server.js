
const express = require('express');
const http = require('http');
const { Server } = require("socket.io");
const redis = require('redis');
const path = require('path');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: "http://localhost:3000", // Allow React dev server
    methods: ["GET", "POST"]
  }
});

const PORT = process.env.PORT || 3001;

// --- Redis Client Setup ---
const redisClient = redis.createClient({
    // assume Redis is on localhost at its default port
    url: 'redis://localhost:6379'
});

const REDIS_CHANNEL = 'bot_dashboard_channel';

redisClient.on('error', (err) => console.log('Redis Client Error', err));

(async () => {
    await redisClient.connect();
    console.log('Connected to Redis successfully.');

    await redisClient.subscribe(REDIS_CHANNEL, (message) => {
        try {
            console.log(`Received message from Redis channel '${REDIS_CHANNEL}':`, message);
            const parsedMessage = JSON.parse(message);
            
            // broadcast messages received from the bot to all frontend clients over WebSocket
            io.emit('bot_update', parsedMessage);

        } catch (e) {
            console.error('Failed to parse or emit message:', e);
        }
    });
    console.log(`Subscribed to Redis channel: ${REDIS_CHANNEL}`);
})();


// --- Socket.IO Connection Handling ---
io.on('connection', (socket) => {
  console.log('A user connected with socket ID:', socket.id);
  
  socket.on('disconnect', () => {
    console.log('User disconnected:', socket.id);
  });
});

// --- Serve React App (for production) ---
// serve the built frontend from the public folder
app.use(express.static(path.join(__dirname, '..', 'frontend', 'build')));

app.get(/^\/(?!api)(?!socket\.io).*/, (req, res) => {
  res.sendFile(path.join(__dirname, '..', 'frontend', 'build', 'index.html'));
});


server.listen(PORT, () => {
  console.log(`Dashboard backend server listening on *:${PORT}`);
});

