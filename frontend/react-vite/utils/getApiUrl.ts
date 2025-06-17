export function getApiUrl() {
  // Check if we're running in Docker
  const isDocker = process.env.NEXT_PUBLIC_DOCKER === 'true';
  
  if (typeof window !== "undefined") {
    if (isDocker) {
      return "http://voice-assistant:8080";
    }
    // Local development
    return "http://localhost:8080";  // Use 8080 to match Docker Compose
  }
  return process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";
}