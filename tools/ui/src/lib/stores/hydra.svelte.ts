/**
 * hydraStore - hydra mode of the chat form and the live state of running heads
 *
 * A hydra request (llama-server, PLAN-hydra.md) has one root, the conversation so far plus the
 * input box, and N heads, each with its own tail message. The server reads the root once and
 * decodes the heads together. Each head is stored as a branch under the root: the tail as a user
 * message and the reply as its child, so any head can be continued as a normal chat.
 */

import { serverStore } from '$lib/stores/server.svelte';
import type { ChatMessagePromptProgress, ChatMessageTimings } from '$lib/types';
import { SvelteMap } from 'svelte/reactivity';

export interface HydraLiveHead {
	content: string;
	reasoning: string;
	done: boolean;
	finishReason?: string;
	timings?: ChatMessageTimings;
	progress?: ChatMessagePromptProgress;
	/** a loop guard reflection runs on this head (llama-server --loop-guard reflect) */
	reflecting?: boolean;
	/** the text of the reflection, shown while it runs */
	reflection?: string;
}

class HydraStore {
	/** the head boxes are hidden while hydra mode stays on (after a send, or by the user) */
	collapsed = $state(false);
	/** hydra mode of the chat form */
	enabled = $state(false);
	/** live state of running heads, keyed by the id of the reply message */
	live = new SvelteMap<string, HydraLiveHead>();
	/** prompt progress of the shared root of the running request, by conversation */
	rootProgress = new SvelteMap<string, ChatMessagePromptProgress>();
	/** one tail per head, edited in the chat form */
	tails = $state<string[]>(['', '']);
	/** thinking of the heads: the chat's own setting, or on / off for every head */
	thinking = $state<'default' | 'on' | 'off'>('default');

	get available(): boolean {
		return this.maxHeads > 0;
	}

	/** most heads the server takes, 0 = the server has no hydra */
	get maxHeads(): number {
		return serverStore.props?.hydra_max_heads ?? 0;
	}

	/** hydra mode is on and every head has a tail */
	get ready(): boolean {
		return (
			this.enabled &&
			this.available &&
			this.tails.length > 0 &&
			this.tails.every((t) => t.trim().length > 0)
		);
	}

	/** a tail for every head: copy the first box into the empty ones */
	fillFromFirst(): void {
		const first = this.tails[0] ?? '';

		this.tails = this.tails.map((t) => (t.trim() ? t : first));
	}

	setCount(n: number): void {
		const count = Math.max(1, Math.min(n, Math.max(1, this.maxHeads)));
		const next = this.tails.slice(0, count);

		while (next.length < count) next.push('');

		this.tails = next;
	}

	updateLive(messageId: string, patch: Partial<HydraLiveHead>): void {
		const cur = this.live.get(messageId) ?? { content: '', done: false, reasoning: '' };

		this.live.set(messageId, { ...cur, ...patch });
	}
}

export const hydraStore = new HydraStore();
