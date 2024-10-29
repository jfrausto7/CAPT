interface MessageRequest {
  text: string;
  conversation_id?: string;
}

export interface Message {
  id: string;
  text: string;
  sender: string;
  timestamp: string;
}

interface MessageResponse {
  conversation_id: string;
  message: Message;
  response: Message;
}

export const sendMessage = async (messageRequest: MessageRequest): Promise<MessageResponse> => {
  try {
    const response = await fetch('http://10.19.212.74:8000/api/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        // Add any auth headers if needed
        // 'Authorization': 'Bearer your-token-here'
      },
      // credentials: '*', // Include cookies if needed
      mode: 'cors', // Explicitly state CORS mode
      body: JSON.stringify(messageRequest),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('API Error:', error);
    throw new Error(error instanceof Error ? error.message : 'Failed to send message');
  }
};