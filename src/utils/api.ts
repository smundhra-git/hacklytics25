const API_URL = process.env.NEXT_PUBLIC_API_URL;

export async function getUserInfo() {
  const token = localStorage.getItem('token');
  if (!token) return null;

  try {
    const response = await fetch(`${API_URL}/api/v1/auth/me`, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });

    if (!response.ok) {
      throw new Error('Failed to fetch user info');
    }

    return await response.json();
  } catch (error) {
    console.error('Error fetching user info:', error);
    return null;
  }
}

export async function logout() {
  localStorage.removeItem('token');
  window.location.href = '/login';
}

export async function queryAssistant(message: string) {
  const token = localStorage.getItem('token');
  if (!token) return null;

  try {
    const response = await fetch(`${API_URL}/api/v1/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({ message })
    });

    if (!response.ok) {
      throw new Error('Failed to query assistant');
    }

    return await response.json();
  } catch (error) {
    console.error('Error querying assistant:', error);
    return null;
  }
} 