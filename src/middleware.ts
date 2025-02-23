import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  const token = request.cookies.get('token')?.value;
  const { pathname } = request.nextUrl;

  // Public paths that don't require authentication
  const publicPaths = ['/', '/login', '/auth/callback', '/features', '/about', '/blog', '/contact', '/demo', '/request-demo'];
  const isPublicPath = publicPaths.some(path => pathname === path);

  // Protected routes that require authentication
  const protectedPaths = ['/dashboard'];
  const isProtectedPath = protectedPaths.some(path => pathname.startsWith(path));

  // Redirect to login if accessing protected route without token
  if (!token && isProtectedPath) {
    const loginUrl = new URL('/login', request.url);
    return NextResponse.redirect(loginUrl);
  }

  // Redirect to dashboard if accessing login page with valid token
  if (token && pathname === '/login') {
    const dashboardUrl = new URL('/dashboard', request.url);
    return NextResponse.redirect(dashboardUrl);
  }

  return NextResponse.next();
}

export const config = {
  matcher: ['/((?!api|_next/static|_next/image|favicon.ico|.*\\.png$).*)'],
}; 