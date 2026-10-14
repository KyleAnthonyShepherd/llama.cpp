import type { OpenAIToolDefinition } from './mcp';

export type HatState = 'ready' | 'stale' | 'unbuilt' | 'unknown';

export interface HatBuildJob {
	running: boolean;
	step: string;
	done: number;
	total: number;
	error: string;
}

export interface HatSummary {
	name: string;
	description: string;
	state: HatState;
	/** why it is not ready */
	why: string[];
	/** prompt tokens of the built hat, null before the first build */
	tokens: number | null;
	/** characters of system text, for an estimate before a build */
	chars: number;
	documents: number;
	tools: number;
	build: HatBuildJob | null;
}

export interface HatDocument {
	file: string;
	reader: string;
	title: string;
	size: number;
	read: boolean;
	chars?: number | null;
	pages?: number | null;
	by_vision?: number | null;
	by_text?: number | null;
	notes?: string[];
}

export interface HatDetail extends HatSummary {
	instructions: string;
	tool_names: string[];
	document_list: HatDocument[];
}

export interface HatServerInfo {
	reachable: boolean;
	model?: string;
	vision?: boolean;
	n_ctx?: number;
	tools?: string[];
}

export interface HatList {
	hats: HatSummary[];
	server: HatServerInfo;
	readers: string[];
	accepts: string[];
}

/** What a chat that wears the hat must send: the system text and tools the server read */
export interface HatWear {
	name: string;
	system: string;
	tools: OpenAIToolDefinition[];
	tokens: number | null;
	state: HatState;
	why: string[];
	/** true when the saved prompt cache was put back */
	restored: boolean;
}

export interface HatSaveFields {
	create?: boolean;
	rename_to?: string;
	description?: string;
	instructions?: string;
	tools?: string[];
	tool_defs?: Record<string, OpenAIToolDefinition>;
	documents?: { file: string; reader: string; title: string }[];
}
