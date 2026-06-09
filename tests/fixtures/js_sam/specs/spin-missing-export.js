/** Fixture: loads fine but violates the module contract (no setState/checkerIntents). */

'use strict';

const state = { lockHeld: false };

module.exports = {
  instance: () => ({ state: () => state }),
  init: () => {},
  actions: { AcquireLock: () => {} },
  getState: () => state,
  // setState missing
  // checkerIntents missing
};
