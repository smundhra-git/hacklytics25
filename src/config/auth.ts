export const authConfig = {
  googleAuth: {
    clientId: process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID,
    redirectUri: process.env.NEXT_PUBLIC_OAUTH_REDIRECT_URI,
    scope: 'openid email profile',
  },
  api: {
    baseUrl: process.env.NEXT_PUBLIC_API_URL,
    endpoints: {
      googleLogin: '/api/v1/auth/login/google',
      googleCallback: '/api/v1/auth/google/callback',
      me: '/api/v1/auth/me',
    }
  },
  routes: {
    login: '/login',
    callback: '/auth/callback',
    dashboard: '/dashboard',
    home: '/',
  }
}; 