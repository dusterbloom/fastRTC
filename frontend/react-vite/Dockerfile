# NOTE: During build, only access to fonts.googleapis.com and fonts.gstatic.com is required for Google Fonts.
# If building in a restricted network, allow outbound HTTPS to these domains.

# Stage 1: Build the app
FROM node:20-alpine AS builder
WORKDIR /app
COPY . .
ENV NEXT_FONT_GOOGLE_FETCH_TIMEOUT=60000
RUN npm install
RUN npm run build

# Stage 2: Run the app
FROM node:20-alpine AS runner
WORKDIR /app
COPY --from=builder /app ./
RUN npm install --production
EXPOSE 3001
CMD ["npm", "run", "start"]