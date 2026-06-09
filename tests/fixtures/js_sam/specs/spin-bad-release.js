/**
 * Fixture: deliberately buggy spin model — ReleaseLock flips lockHeld but
 * never clears lockHolder or the thread status. Phase 3 must fail exactly
 * the windows that exercise the release path; Phase 4 LockStatusConsistency
 * must produce a counterexample.
 */

'use strict';

const { createInstance } = require('@cognitive-fab/sam-pattern');

const instance = createInstance({ instanceName: 'spin-bad', hasAsyncActions: false });

const INITIAL_STATE = {
  lockHeld: false,
  lockHolder: null,
  threadStatus: { 0: 'idle', 1: 'idle' },
  callType: { 0: null, 1: null },
};

const clone = (value) => JSON.parse(JSON.stringify(value));

const { intents } = instance({
  initialState: clone(INITIAL_STATE),
  component: {
    actions: [
      ({ thread, callType = 'lock' } = {}) => ({ __name: 'AcquireLock', acquire: { thread, callType } }),
      ({ thread } = {}) => ({ __name: 'ReleaseLock', release: { thread } }),
    ],
    acceptors: [
      (model) => ({ acquire }) => {
        if (acquire == null) return;
        const { thread, callType } = acquire;
        const id = String(thread);
        if (model.threadStatus[id] === undefined || model.threadStatus[id] === 'locked') return;
        if (!model.lockHeld) {
          model.lockHeld = true;
          model.lockHolder = thread;
          model.threadStatus[id] = 'locked';
          model.callType[id] = null;
        } else if (callType === 'tryLock') {
          model.threadStatus[id] = 'idle';
          model.callType[id] = null;
        } else {
          model.threadStatus[id] = 'trying';
          model.callType[id] = 'lock';
        }
      },
      (model) => ({ release }) => {
        if (release == null) return;
        if (model.lockHolder !== release.thread) return;
        // BUG: lock is freed but holder/status bookkeeping is never cleared.
        model.lockHeld = false;
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
  {
    name: 'AcquireLock',
    intent: actions.AcquireLock,
    values: [
      [{ thread: 0, callType: 'lock' }],
      [{ thread: 1, callType: 'lock' }],
      [{ thread: 0, callType: 'tryLock' }],
      [{ thread: 1, callType: 'tryLock' }],
    ],
  },
  {
    name: 'ReleaseLock',
    intent: actions.ReleaseLock,
    values: [[{ thread: 0 }], [{ thread: 1 }]],
  },
];

module.exports = { instance, init, actions, getState, setState, checkerIntents };
