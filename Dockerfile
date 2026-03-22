# Base image
FROM node:18

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install Node dependencies
WORKDIR /app/server
RUN npm install

# Install Python dependencies
WORKDIR /app
RUN pip3 install pandas scikit-learn numpy

# Expose port
EXPOSE 3000

# Start server
WORKDIR /app/server
CMD ["node", "index.js"]