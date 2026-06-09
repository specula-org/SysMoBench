/**
 * JS-SAM evaluation helper.
 *
 * The Python backend (tla_eval/languages/js_sam.py) shells out to this CLI
 * with one subcommand per evaluation phase. Every subcommand reads a single
 * JSON request on stdin and writes a single JSON response on stdout.
 *
 * Exit code 0 means "the helper ran"; pass/fail is carried inside the JSON
 * (mirroring the PAT backend, whose console always exits 0). A non-zero exit
 * or malformed JSON is a tooling failure, not a spec failure.
 *
 * Subcommands:
 *   ping         — tool diagnostics ({} -> { ok, node, samPattern })
 *   validate     — Phase 1: parse + load + structural contract check
 *   check        — Phase 2: bounded behavior exploration (sam-pattern checker)
 *   transitions  — Phase 3: replay (pre, action, post) trace windows
 *   invariants   — Phase 4: explore with translated safety predicates
 *
 * Generated specs are CommonJS modules that must export:
 *   { instance, init, actions, getState, setState, checkerIntents }
 * See tla_eval/tasks/<task>/prompts/js-sam/direct_call.txt for the contract.
 */

import { createRequire } from 'node:module';
import Module from 'node:module';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const HELPER_DIR = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

// Generated specs live in arbitrary work dirs, so their
// require('@cognitive-fab/sam-pattern') would not resolve by walking up from
// the spec file. Append the helper's node_modules to the global resolution
// paths before any spec is loaded.
process.env.NODE_PATH = [path.join(HELPER_DIR, 'node_modules'), process.env.NODE_PATH]
  .filter(Boolean)
  .join(path.delimiter);
Module._initPaths();

const { checker } = require('@cognitive-fab/sam-pattern');

const DEFAULT_DEPTH_MAX = 6;
const MAX_REPORTED_VIOLATIONS = 5;
const MAX_REPORTED_WINDOW_FAILURES = 20;

// SAM rejects gated intents by presenting { __error: 'unexpected action …' }.
// That is legal model behavior (a rejected action), not a runtime fault.
const REJECTION_PREFIX = 'unexpected action';

// sam-pattern invokes acceptors through an async wrapper, so an acceptor
// throw surfaces as an unhandled promise rejection (which would kill this
// process) instead of a synchronous exception. Collect them and let each
// subcommand drain the queue at a point where it can attribute the failure.
const pendingAsyncErrors = [];
process.on('unhandledRejection', (reason) => {
  if (pendingAsyncErrors.length < 50) {
    pendingAsyncErrors.push(reason instanceof Error ? reason.message : String(reason));
  }
});

/** Lets queued microtasks/timers settle, then returns the captured errors. */
const drainAsyncErrors = async () => {
  await new Promise((resolve) => setTimeout(resolve, 0));
  return pendingAsyncErrors.splice(0);
};

/** Strips SAM-internal (__-prefixed) keys and functions from a state tree. */
const sanitizeReplacer = (key, value) => {
  if (typeof key === 'string' && key.startsWith('__')) return undefined;
  if (typeof value === 'function') return undefined;
  return value;
};

/** @returns {object} a plain JSON snapshot of a live SAM model/state. */
const plainSnapshot = (state) => JSON.parse(JSON.stringify(state, sanitizeReplacer));

const isRejectionError = (rawError) => {
  const text = typeof rawError === 'string' ? rawError : (rawError && rawError.message) || '';
  return text.startsWith(REJECTION_PREFIX);
};

/** Reads the model's error slot through the module's exported instance. */
const readModelError = (mod) => {
  const model = mod.instance({}).state();
  if (!model.hasError()) return null;
  const raw = model.error();
  const message = model.errorMessage() ?? String(raw ?? 'unknown error');
  return { raw, message, isRejection: isRejectionError(raw) };
};

const clearModelError = (mod) => {
  mod.instance({}).state().clearError();
};

// ---------------------------------------------------------------------------
// Spec loading + structural contract check
// ---------------------------------------------------------------------------

/**
 * Loads a generated spec and verifies the module contract.
 *
 * @param {string} specPath absolute path to the generated .js module
 * @param {string[]} targetActions mandatory action names ([] to skip)
 * @returns {{ mod?: object, syntaxErrors: string[], semanticErrors: string[] }}
 */
const loadSpec = (specPath, targetActions = []) => {
  const syntaxErrors = [];
  const semanticErrors = [];
  let mod;

  try {
    mod = require(path.resolve(specPath));
  } catch (err) {
    const description = `${err.name ?? 'Error'}: ${err.message}`;
    if (err instanceof SyntaxError) {
      syntaxErrors.push(description);
    } else {
      semanticErrors.push(`Module failed to load: ${description}`);
    }
    return { syntaxErrors, semanticErrors };
  }

  if (mod == null || typeof mod !== 'object') {
    semanticErrors.push('module.exports is not an object');
    return { syntaxErrors, semanticErrors };
  }

  for (const fn of ['init', 'getState', 'setState', 'instance']) {
    if (typeof mod[fn] !== 'function') {
      semanticErrors.push(`Missing or non-function export: ${fn}`);
    }
  }

  if (mod.actions == null || typeof mod.actions !== 'object') {
    semanticErrors.push('Missing export: actions (object mapping action name -> function)');
  } else {
    for (const [name, fn] of Object.entries(mod.actions)) {
      if (typeof fn !== 'function') {
        semanticErrors.push(`actions.${name} is not a function`);
      }
    }
    for (const name of targetActions) {
      if (typeof mod.actions[name] !== 'function') {
        semanticErrors.push(`Mandatory target action missing from actions: ${name}`);
      }
    }
  }

  if (!Array.isArray(mod.checkerIntents) || mod.checkerIntents.length === 0) {
    semanticErrors.push('Missing export: checkerIntents (non-empty array of { name, intent, values })');
  } else {
    mod.checkerIntents.forEach((descriptor, index) => {
      if (descriptor == null || typeof descriptor !== 'object') {
        semanticErrors.push(`checkerIntents[${index}] is not an object`);
        return;
      }
      if (typeof descriptor.name !== 'string') {
        semanticErrors.push(`checkerIntents[${index}].name is not a string`);
      }
      if (typeof descriptor.intent !== 'function') {
        semanticErrors.push(`checkerIntents[${index}].intent is not a function`);
      }
      if (!Array.isArray(descriptor.values) || descriptor.values.length === 0
          || !descriptor.values.every(Array.isArray)) {
        semanticErrors.push(
          `checkerIntents[${index}].values must be a non-empty array of argument arrays `
          + '(use [[]] for an action with no arguments)',
        );
      }
    });
  }

  if (semanticErrors.length > 0) {
    return { syntaxErrors, semanticErrors };
  }

  // Smoke checks: init/getState must work and be deterministic + serializable.
  try {
    mod.init();
    const first = JSON.stringify(mod.getState());
    mod.init();
    const second = JSON.stringify(mod.getState());
    if (typeof mod.getState() !== 'object' || mod.getState() == null) {
      semanticErrors.push('getState() did not return an object');
    } else if (first !== second) {
      semanticErrors.push('init() + getState() is not deterministic across two runs');
    }
  } catch (err) {
    semanticErrors.push(`init()/getState() smoke check threw: ${err.message}`);
  }

  return { mod, syntaxErrors, semanticErrors };
};

// ---------------------------------------------------------------------------
// Bounded exploration (shared by `check` and `invariants`)
// ---------------------------------------------------------------------------

/**
 * Runs the sam-pattern behavior explorer over all intent permutations.
 *
 * @param {object} mod loaded spec module
 * @param {number} depthMax exploration depth bound
 * @param {Array<{name: string, holds: (state: object) => boolean}>} predicates
 *        safety predicates that must hold in every reachable state
 * @returns {{ steps: number, runtimeErrors: object[], violations: object[],
 *             predicateErrors: object[], serializationErrors: string[] }}
 */
const explore = (mod, depthMax, predicates = []) => {
  const runtimeErrors = [];
  const violations = [];
  const predicateErrors = [];
  const serializationErrors = [];
  let steps = 0;

  const safety = (state) => {
    steps += 1;

    // Unhandled exceptions surface in the model's error slot. Rejected
    // (gated) actions use the same slot but are legal behavior.
    if (state.hasError()) {
      const raw = state.error();
      const message = state.errorMessage() ?? String(raw ?? 'unknown error');
      const rejection = isRejectionError(raw);
      state.clearError();
      if (!rejection && runtimeErrors.length < MAX_REPORTED_VIOLATIONS) {
        runtimeErrors.push({ message, behavior: [...(state.__behavior ?? [])] });
      }
      if (!rejection) return false; // recorded separately, not as a predicate violation
    }

    let snapshot;
    try {
      snapshot = plainSnapshot(state);
    } catch (err) {
      if (serializationErrors.length < MAX_REPORTED_VIOLATIONS) {
        serializationErrors.push(`State is not JSON-serializable: ${err.message}`);
      }
      return false;
    }

    for (const predicate of predicates) {
      let holds;
      try {
        holds = predicate.holds(snapshot);
      } catch (err) {
        if (predicateErrors.length < MAX_REPORTED_VIOLATIONS) {
          predicateErrors.push({ name: predicate.name, message: err.message });
        }
        continue;
      }
      if (!holds) return true; // checker records the violating behavior
    }
    return false;
  };

  clearModelError(mod);
  mod.init();
  checker(
    {
      instance: mod.instance,
      intents: mod.checkerIntents,
      reset: () => mod.init(),
      safety,
      options: { depthMax },
    },
    () => {},
    (behavior) => {
      if (violations.length < MAX_REPORTED_VIOLATIONS) {
        violations.push({ behavior: Array.isArray(behavior) ? [...behavior] : behavior });
      }
    },
  );

  return { steps, runtimeErrors, violations, predicateErrors, serializationErrors };
};

// ---------------------------------------------------------------------------
// Subcommands
// ---------------------------------------------------------------------------

const cmdPing = () => ({
  ok: true,
  node: process.version,
  samPattern: require('@cognitive-fab/sam-pattern/package.json').version,
  samFsm: require('@cognitive-fab/sam-fsm/package.json').version,
});

const cmdValidate = (request) => {
  const { specPath, targetActions = [] } = request;
  const { syntaxErrors, semanticErrors } = loadSpec(specPath, targetActions);
  return {
    ok: true,
    success: syntaxErrors.length === 0 && semanticErrors.length === 0,
    syntaxErrors,
    semanticErrors,
  };
};

const cmdCheck = async (request) => {
  const { specPath, depthMax = DEFAULT_DEPTH_MAX, targetActions = [] } = request;
  const { mod, syntaxErrors, semanticErrors } = loadSpec(specPath, targetActions);
  if (!mod) {
    return {
      ok: true,
      success: false,
      classification: 'parse_error',
      errors: [...syntaxErrors, ...semanticErrors],
    };
  }

  const started = Date.now();
  let first;
  let second;
  try {
    const dedupe = (messages) => [...new Set(messages)].slice(0, MAX_REPORTED_VIOLATIONS);
    first = explore(mod, depthMax);
    first.runtimeErrors.push(...dedupe(await drainAsyncErrors()).map((message) => ({ message, behavior: [] })));
    // Second pass over the identical exploration: any digest difference means
    // the spec depends on nondeterminism (randomness, time, ambient state).
    second = explore(mod, depthMax);
    second.runtimeErrors.push(...dedupe(await drainAsyncErrors()).map((message) => ({ message, behavior: [] })));
  } catch (err) {
    return {
      ok: true,
      success: false,
      classification: 'runtime_error',
      errors: [`Behavior exploration aborted: ${err.message}`],
      elapsedMs: Date.now() - started,
    };
  }
  const elapsedMs = Date.now() - started;

  const digest = (run) => JSON.stringify([run.steps, run.runtimeErrors, run.serializationErrors]);
  const isDeterministic = digest(first) === digest(second);

  let classification = null;
  const errors = [];
  if (first.runtimeErrors.length > 0) {
    classification = 'runtime_error';
    errors.push(...new Set(first.runtimeErrors.map((e) => e.message)));
  } else if (first.serializationErrors.length > 0) {
    classification = 'runtime_error';
    errors.push(...first.serializationErrors);
  } else if (!isDeterministic) {
    classification = 'nondeterminism';
    errors.push('Two identical explorations produced different outcomes.');
  }

  return {
    ok: true,
    success: classification === null,
    classification,
    depthMax,
    stepsExplored: first.steps,
    runtimeErrors: first.runtimeErrors,
    errors,
    elapsedMs,
  };
};

/**
 * Deep projection diff: every key present in `expected` must be (deeply)
 * equal in `actual`. Keys present only in `actual` are model-internal
 * bookkeeping and are ignored. `null` and `undefined` are equivalent.
 *
 * @returns {string[]} mismatch descriptions (empty when the window matches)
 */
const projectionDiff = (expected, actual, prefix = '') => {
  if (expected === null || expected === undefined) {
    return actual === null || actual === undefined
      ? []
      : [`${prefix || '$'}: expected null, got ${JSON.stringify(actual)}`];
  }
  if (typeof expected !== 'object') {
    return Object.is(expected, actual)
      ? []
      : [`${prefix || '$'}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`];
  }
  if (Array.isArray(expected)) {
    if (!Array.isArray(actual) || actual.length !== expected.length) {
      return [`${prefix || '$'}: expected array ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`];
    }
    return expected.flatMap((item, i) => projectionDiff(item, actual[i], `${prefix}[${i}]`));
  }
  if (actual === null || actual === undefined || typeof actual !== 'object') {
    return [`${prefix || '$'}: expected object, got ${JSON.stringify(actual)}`];
  }
  return Object.keys(expected).flatMap((key) => (
    projectionDiff(expected[key], actual[key], prefix ? `${prefix}.${key}` : key)
  ));
};

const cmdTransitions = async (request) => {
  const { specPath, windows = [], targetActions = [] } = request;
  const { mod, syntaxErrors, semanticErrors } = loadSpec(specPath, targetActions);
  if (!mod) {
    return {
      ok: true,
      success: false,
      error: `Spec failed to load: ${[...syntaxErrors, ...semanticErrors].join('; ')}`,
    };
  }

  const perAction = {};
  const failures = [];
  let totalPassed = 0;

  const record = (action, passed, windowIndex, reason) => {
    const bucket = perAction[action] ?? (perAction[action] = { passed: 0, total: 0 });
    bucket.total += 1;
    if (passed) {
      bucket.passed += 1;
      totalPassed += 1;
    } else if (failures.length < MAX_REPORTED_WINDOW_FAILURES) {
      failures.push({ window: windowIndex, action, reason });
    }
  };

  for (const [index, window] of windows.entries()) {
    const { action, data = {}, preState, postState } = window;
    try {
      const handler = mod.actions[action];
      if (typeof handler !== 'function') {
        record(action, false, index, `action '${action}' is not exported by the spec`);
        continue;
      }
      mod.init();
      clearModelError(mod);
      mod.setState(preState);
      handler(data);

      // Acceptor exceptions surface asynchronously; settle them before diffing
      // so the failure is attributed to this window.
      const asyncErrors = await drainAsyncErrors();
      if (asyncErrors.length > 0) {
        record(action, false, index, `action threw: ${asyncErrors[0]}`);
        continue;
      }

      const modelError = readModelError(mod);
      if (modelError) {
        clearModelError(mod);
        const kind = modelError.isRejection ? 'rejected by the model' : 'threw';
        record(action, false, index, `action ${kind}: ${modelError.message}`);
        continue;
      }

      const mismatches = projectionDiff(postState, mod.getState());
      record(action, mismatches.length === 0, index, mismatches.slice(0, 5).join('; '));
    } catch (err) {
      record(action, false, index, `replay raised: ${err.message}`);
    }
  }

  for (const bucket of Object.values(perAction)) {
    bucket.rate = bucket.total > 0 ? bucket.passed / bucket.total : 0;
  }

  return {
    ok: true,
    success: true,
    totalPassed,
    totalWindows: windows.length,
    perAction,
    failures,
  };
};

const cmdInvariants = async (request) => {
  const { specPath, invariants = [], depthMax = DEFAULT_DEPTH_MAX, targetActions = [] } = request;
  const { mod, syntaxErrors, semanticErrors } = loadSpec(specPath, targetActions);
  if (!mod) {
    return {
      ok: true,
      success: false,
      error: `Spec failed to load: ${[...syntaxErrors, ...semanticErrors].join('; ')}`,
    };
  }

  const results = [];
  for (const { name, predicate } of invariants) {
    const started = Date.now();
    let holds;
    try {
      // The translated predicate is a JS arrow/function expression:
      // (state) => boolean, true when the invariant HOLDS.
      holds = (0, eval)(`(${predicate})`);
      if (typeof holds !== 'function') {
        throw new TypeError('predicate did not evaluate to a function');
      }
    } catch (err) {
      results.push({
        name,
        success: false,
        error: `Predicate failed to compile: ${err.message}`,
        elapsedMs: Date.now() - started,
      });
      continue;
    }

    let run;
    try {
      run = explore(mod, depthMax, [{ name, holds }]);
      run.runtimeErrors.push(...(await drainAsyncErrors()).map((message) => ({ message, behavior: [] })));
    } catch (err) {
      results.push({
        name,
        success: false,
        error: `Exploration aborted: ${err.message}`,
        elapsedMs: Date.now() - started,
      });
      continue;
    }

    const problems = [];
    if (run.violations.length > 0) {
      problems.push(`invariant violated in ${run.violations.length}+ behavior(s)`);
    }
    if (run.predicateErrors.length > 0) {
      problems.push(`predicate threw: ${run.predicateErrors[0].message}`);
    }
    if (run.runtimeErrors.length > 0) {
      problems.push(`spec raised during exploration: ${run.runtimeErrors[0].message}`);
    }

    results.push({
      name,
      success: problems.length === 0,
      error: problems.length > 0 ? problems.join('; ') : null,
      counterexample: run.violations[0]?.behavior ?? null,
      stepsExplored: run.steps,
      elapsedMs: Date.now() - started,
    });
  }

  return { ok: true, success: true, results };
};

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

const COMMANDS = {
  ping: cmdPing,
  validate: cmdValidate,
  check: cmdCheck,
  transitions: cmdTransitions,
  invariants: cmdInvariants,
};

const readStdin = async () => {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  return Buffer.concat(chunks).toString('utf-8');
};

const main = async () => {
  const subcommand = process.argv[2];
  const handler = COMMANDS[subcommand];
  if (!handler) {
    process.stdout.write(JSON.stringify({
      ok: false,
      error: `Unknown subcommand '${subcommand}'. Expected one of: ${Object.keys(COMMANDS).join(', ')}`,
    }));
    process.exitCode = 2;
    return;
  }

  let request = {};
  if (subcommand !== 'ping') {
    const raw = await readStdin();
    try {
      request = raw.trim() ? JSON.parse(raw) : {};
    } catch (err) {
      process.stdout.write(JSON.stringify({ ok: false, error: `Invalid JSON request: ${err.message}` }));
      process.exitCode = 2;
      return;
    }
  }

  try {
    const response = await handler(request);
    process.stdout.write(JSON.stringify(response));
  } catch (err) {
    process.stdout.write(JSON.stringify({ ok: false, error: `${err.name}: ${err.message}`, stack: err.stack }));
    process.exitCode = 1;
  }
};

await main();
