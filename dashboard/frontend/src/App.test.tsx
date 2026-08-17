import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';

// App opens a socket in useEffect; stub the client so tests stay offline
// and deterministic rather than attempting a real connection from jsdom.
jest.mock('socket.io-client', () => ({
  __esModule: true,
  default: () => ({ on: jest.fn(), disconnect: jest.fn() }),
}));

test('renders the dashboard heading', () => {
  render(<App />);
  expect(screen.getByText(/Arbitrage Bot Dashboard/i)).toBeInTheDocument();
});

test('starts in the connecting state until the backend responds', () => {
  render(<App />);
  expect(screen.getByText('CONNECTING')).toBeInTheDocument();
  expect(screen.getByText(/Connecting to dashboard backend/i)).toBeInTheDocument();
});

test('renders the three dashboard panels', () => {
  const { container } = render(<App />);
  expect(screen.getByText('Bot Status')).toBeInTheDocument();
  expect(screen.getByText('Activity Log')).toBeInTheDocument();
  // Chart.js paints its title onto a canvas, so assert the canvas exists
  // rather than querying for text that never reaches the DOM.
  expect(container.querySelector('canvas')).toBeInTheDocument();
});
