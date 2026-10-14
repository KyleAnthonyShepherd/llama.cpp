import { STORAGE_APP_NAME } from './storage.constants';

// A hat is a system prompt, a tool list and documents that the model has already read: the server
// keeps the resulting prompt cache on disk and restores it when a chat starts.
// The hats tool lists, edits, builds and restores them. The UI calls it, it is never offered to the model.
export const HATS_TOOL_SUFFIX = 'hats_ui';

/** Hat picked for new chats */
export const HAT_DEFAULT_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME}.hatDefault`;

export const HAT_BUILD_POLL_MS = 1500;

export const HAT_DOCUMENT_MAX_BYTES = 64 * 1024 * 1024;

export const HAT_NAME_PATTERN = /^[a-z0-9][a-z0-9_-]{0,39}$/;

export const HAT_STATE_LABELS: Record<string, string> = {
	ready: 'ready',
	stale: 'needs a rebuild',
	unbuilt: 'not built',
	unknown: 'unknown'
};

export const HAT_READER_LABELS: Record<string, string> = {
	auto: 'Auto (text, vision where needed)',
	text: 'Text only',
	vision: 'Vision (every page)'
};
