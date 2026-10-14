/**
 * hatsStore - Hats offered by the server, and the one a chat wears
 *
 * A hat is a system prompt, a tool list and documents the model has already
 * read. The server keeps the prompt cache it built from them on disk. Wearing
 * a hat restores that cache and returns the system text and tool definitions
 * it was built from: a chat that sends exactly those starts with nothing to
 * process. Which hat a chat wears lives on the conversation row
 * (conversationsStore.preferences).
 */

import { MessageRole } from '$lib/enums';
import { HatsService } from '$lib/services/hats.service';
// direct imports between stores, not via the barrel, to avoid circular deps
import { toolsStore } from '$lib/stores/tools.svelte';
import type {
	ApiChatMessageData,
	HatDetail,
	HatSaveFields,
	HatServerInfo,
	HatSummary,
	HatWear
} from '$lib/types';
import { toast } from 'svelte-sonner';

class HatsStore {
	private _accepts = $state<string[]>([]);
	private _error = $state<string | null>(null);
	private _hats = $state<HatSummary[]>([]);
	private _loading = $state(false);
	private _readers = $state<string[]>([]);
	private _server = $state<HatServerInfo | null>(null);
	// the server has one prompt cache to restore into: a request for another chat displaces it
	private lastConversationId: string | null = null;
	private warnedMissing = false;
	private worn = new Map<string, HatWear>();

	get accepts(): string[] {
		return this._accepts;
	}

	/** True when the server offers a hats tool */
	get available(): boolean {
		return toolsStore.hatsTool !== null;
	}

	get error(): string | null {
		return this._error;
	}

	get hats(): HatSummary[] {
		return this._hats;
	}

	get loading(): boolean {
		return this._loading;
	}

	get readers(): string[] {
		return this._readers;
	}

	get server(): HatServerInfo | null {
		return this._server;
	}

	/**
	 * The hat's system text in place of the chat's own system messages.
	 * The server cache only matches a prompt that starts exactly as the hat does.
	 */
	applyToMessages(hat: HatWear, messages: ApiChatMessageData[]): ApiChatMessageData[] {
		const rest = messages.filter((m) => m.role !== MessageRole.SYSTEM);

		if (!hat.system) return rest;

		return [{ content: hat.system, role: MessageRole.SYSTEM }, ...rest];
	}

	async build(name: string): Promise<void> {
		this.upsert(await HatsService.build(this.tool(), name));
		this.worn.delete(name);
	}

	async cancel(name: string): Promise<void> {
		await HatsService.cancel(this.tool(), name);
	}

	find(name: string | null): HatSummary | null {
		return name ? (this._hats.find((h) => h.name === name) ?? null) : null;
	}

	async get(name: string): Promise<HatDetail> {
		return HatsService.get(this.tool(), name);
	}

	/**
	 * Restore the hat's prompt cache when this request could not otherwise
	 * hit it, and return what the request must send.
	 *
	 * @throws if the hat cannot be worn, e.g. it has documents nobody has read yet
	 */
	async prepare(conversationId: string, name: string, signal?: AbortSignal): Promise<HatWear> {
		const cached = this.worn.get(name);

		if (cached && this.lastConversationId === conversationId) return cached;

		const hat = await HatsService.wear(this.tool(), name, signal);

		this.worn.set(name, hat);
		this.lastConversationId = conversationId;

		if (!hat.restored) {
			toast.warning(
				`Hat "${name}" has no saved cache for this model (${hat.why.join('; ') || hat.state}). This message will take longer. Build the hat to fix that.`,
				{ duration: 10000 }
			);
		}

		return hat;
	}

	async putDocument(name: string, file: File): Promise<HatDetail> {
		this.worn.delete(name);

		return HatsService.putDocument(this.tool(), name, file);
	}

	async refresh(): Promise<void> {
		if (!this.available || this._loading) return;

		this._loading = true;

		try {
			const listed = await HatsService.list(this.tool());

			this._accepts = listed.accepts;
			this._error = null;
			this._hats = listed.hats;
			this._readers = listed.readers;
			this._server = listed.server;
		} catch (err) {
			this._error = err instanceof Error ? err.message : String(err);
		} finally {
			this._loading = false;
		}
	}

	async remove(name: string): Promise<void> {
		await HatsService.remove(this.tool(), name);
		this.worn.delete(name);
		this._hats = this._hats.filter((h) => h.name !== name);
	}

	async removeDocument(name: string, file: string): Promise<HatDetail> {
		this.worn.delete(name);

		return HatsService.removeDocument(this.tool(), name, file);
	}

	async save(name: string, fields: HatSaveFields): Promise<HatDetail> {
		const saved = await HatsService.save(this.tool(), name, fields);

		this.worn.delete(name);
		this.worn.delete(saved.name);
		await this.refresh();

		return saved;
	}

	async status(name: string): Promise<HatSummary> {
		const status = await HatsService.status(this.tool(), name);

		this.upsert(status);

		return status;
	}

	/** One warning per session when a chat names a hat this server does not have */
	warnMissing(name: string): void {
		if (this.warnedMissing) return;

		this.warnedMissing = true;
		toast.warning(
			`This chat wears the hat "${name}", but the server offers no hats. It is sent without one.`,
			{ duration: 10000 }
		);
	}

	private tool(): string {
		const tool = toolsStore.hatsTool;

		if (!tool) throw new Error('the server offers no hats tool');

		return tool;
	}

	private upsert(status: HatSummary): void {
		const index = this._hats.findIndex((h) => h.name === status.name);

		if (index === -1) this._hats = [...this._hats, status];
		else this._hats[index] = status;
	}
}

export const hatsStore = new HatsStore();
