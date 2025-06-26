# frontend/react-vite/Dockerfile - Frontend container with WebRTC environment
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Copy source code (including .env.docker files)
COPY . .

# Accept build arguments for Next.js environment variables
ARG NEXT_PUBLIC_API_URL
ARG NEXT_PUBLIC_WEBRTC_API_URL
ARG NEXT_PUBLIC_STUN_SERVERS
ARG NEXT_PUBLIC_TURN_SERVERS

# Build for production with WebRTC support
# Next.js will read from .env.docker and build args
RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]