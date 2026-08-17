// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
// learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom';

// jsdom does not implement scrollIntoView, which ActivityLog calls to keep the
// newest entry visible. Stub it so component tests can render the real tree.
window.HTMLElement.prototype.scrollIntoView = jest.fn();
