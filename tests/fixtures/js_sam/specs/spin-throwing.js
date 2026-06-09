/**
 * Fixture: structurally valid spec whose ReleaseLock acceptor throws on a
 * reachable state. Phase 2 exploration must classify this as runtime_error.
 */

'use strict';

const { createInstance } = require('@cognitive-fab/sam-pattern');

const instance = createInstance({ instanceName: 'spin-throwing', hasAsyncActions: false });

const INITIAL_STATE = {
  lockHeld: false,
  lockHolder: null,
  threadStatus: { 0: 'idle', 1: 'idle' },
};

const clone = (value) => JSON.parse(JSON.stringify(value));

const { intents } = instance({
  initialState: clone(INITIAL_STATE),
  component: {
    actions: [
      ({ thread } = {}) => ({ __name: 'AcquireLock', acquire: { thread } }),
      ({ thread } = {}) => ({ __name: 'ReleaseLock', release: { thread } }),
    ],
    acceptors: [
      (model) => ({ acquire }) => {
        if (acquire == null) return;
        const id = String(acquire.thread);
        if (model.threadStatus[id] === undefined || model.lockHeld) return;
        model.lockHeld = true;
        model.lockHolder = acquire.thread;
        model.threadStatus[id] = 'locked';
      },
      (model) => ({ release }) => {
        if (release == null) return;
        if (model.lockHolder !== release.thread) {
          // BUG: a release by a non-holder is a reachable state during
          // exploration; a robust model would reject it as a no-op.
          throw new Error('release by non-holder');
        }
        model.lockHeld = false;
        model.lockHolder = null;
        model.threadStatus[String(release.thread)] = 'idle';
      },
    ],
  },
});

const [acquireIntent, releaseIntent] = intents;

const sanitizeReplacer = (key, value) => {
  if (typeof key === 'string' && key.startsWith('__')) return undefined;
  if (typeof value === 'function') return undefined;
  return value;
};

const getState = () => JSON.parse(JSON.stringify(instance({}).state(), sanitizeReplacer));

const setState = (snapshot) => {
  instance({ initialState: clone(snapshot) });
};

const init = () => {
  instance({}).state().clearError();
  setState(INITIAL_STATE);
};

const actions = {
  AcquireLock: (data = {}) => acquireIntent(data),
  ReleaseLock: (data = {}) => releaseIntent(data),
};

const checkerIntents = [
  { name: 'AcquireLock', intent: actions.AcquireLock, values: [[{ thread: 0 }], [{ thread: 1 }]] },
  { name: 'ReleaseLock', intent: actions.ReleaseLock, values: [[{ thread: 0 }], [{ thread: 1 }]] },
];

module.exports = { instance, init, actions, getState, setState, checkerIntents };
