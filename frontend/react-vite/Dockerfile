# frontend/react-vite/Dockerfile - Frontend container with WebRTC environment
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Copy source code
COPY . .

# Build for production with WebRTC support
RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]