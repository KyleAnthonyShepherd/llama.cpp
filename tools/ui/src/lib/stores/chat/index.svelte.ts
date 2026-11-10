/**
 * chatStore - Chat lifecycle, streaming and message operations
 *
 * Owns the active conversation's chat state: sending messages, streaming
 * responses, editing/regeneration flows and per-conversation processing
 * activity. Composes the stream manager, message flows, activity ledger and
 * processing snapshot; persists through conversationsStore.
 *
 * Uses ChatService for the API layer and conversationsStore for persistence.
 */

import {
	COMPACTION,
	CWD_CLEARED_TEXT,
	REASONING_EFFORT_TOKENS,
	SYSTEM_MESSAGE_PLACEHOLDER,
	TITLE_GENERATION
} from '$lib/constants';
import {
	ErrorDialogType,
	MessageRole,
	MessageType,
	ReasoningEffort,
	StreamConnectionState
} from '$lib/enums';
import { ChatService } from '$lib/services/chat.service';
import { DatabaseService } from '$lib/services/database.service';
// direct imports between stores, not via the barrel, to avoid circular deps
import { agenticStore } from '$lib/stores/agentic/index.svelte';
import { chatActivityStore } from '$lib/stores/chat/activity.svelte';
import { type ChatFlowsHost, ChatMessageFlows } from '$lib/stores/chat/flows.svelte';
import { chatProcessingStore } from '$lib/stores/chat/processing.svelte';
import { type ChatStreamHost, ChatStreamManager } from '$lib/stores/chat/streams.svelte';
import { conversationsStore } from '$lib/stores/conversations/index.svelte';
import { hatsStore } from '$lib/stores/hats.svelte';
import { hydraStore } from '$lib/stores/hydra.svelte';
import { mcpStore } from '$lib/stores/mcp/index.svelte';
import { modelsStore } from '$lib/stores/models/index.svelte';
import { serverStore } from '$lib/stores/server.svelte';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import { toolsStore } from '$lib/stores/tools.svelte';
import type {
	ApiChatMessageData,
	ChatMessageCompactionTimings,
	ChatMessagePromptProgress,
	ChatMessageTimings,
	ChatStreamCallbacks,
	DatabaseConversation,
	DatabaseMessage,
	DatabaseMessageExtra,
	DatabaseMessageHydra,
	ErrorDialogState,
	HatWear
} from '$lib/types';
import type { ApiLoopGuardEvent } from '$lib/types/api';
import type { AgenticToolPolicy } from '$lib/types/agentic';
import {
	canCompactMessages,
	compactionCut,
	filterByLeafNodeId,
	findMessageById,
	formatCwdMessage,
	getConversationModel,
	isAbortError,
	normalizeModelName,
	sliceAtLastCompaction,
	uuid
} from '$lib/utils';
import { SvelteMap, SvelteSet } from 'svelte/reactivity';
import { toast } from 'svelte-sonner';

class ChatStore implements ChatStreamHost, ChatFlowsHost {
	chatReasoningStates = new SvelteMap<string, boolean>();
	chatStreamingStates = new SvelteMap<
		string,
		{ response: string; messageId: string; model?: string | null }
	>();
	errorDialogState = $state<ErrorDialogState | null>(null);
	// true while the active conversation has a local pipe (send, attach or resume-wait)
	isLoading = $derived(this.activity.isLocal(conversationsStore.activeConversation?.id ?? ''));
	// true while the active conversation streams reasoning content but no visible content yet
	isReasoning = $derived(
		this.chatReasoningStates.get(conversationsStore.activeConversation?.id ?? '') ?? false
	);
	pendingEditMessageId = $state<string | null>(null);
	// resumable stream connection state for the active conversation
	// streaming -> bytes flowing normally, resuming -> waiting on /v1/stream reconnect, lost -> unrecoverable
	streamConnectionState = $state<StreamConnectionState>(StreamConnectionState.STREAMING);
	private abortControllers = new SvelteMap<string, AbortController>();
	private addFilesHandler: ((files: File[]) => void) | null = $state(null);
	// message flows: edit, regenerate, continue, delete
	private flows = new ChatMessageFlows(this);
	private isEditModeActive = $state(false);
	/** Reactive: queued pending messages for non-agentic streaming */
	private pendingMessages = new SvelteMap<
		string,
		{ content: string; extras?: DatabaseMessageExtra[] }
	>();
	private preEncodeAbortController: AbortController | null = null;
	// server-side stream sessions: discovery, attach/replay, resume retry, remote sync
	private streams = new ChatStreamManager(this);

	private turnPrefill: { reasoning: string; response: string } | null = null;

	/** Conv activity (local pipe / remote session), composed here. */
	get activity() {
		return chatActivityStore;
	}

	/** Processing state, composed here so consumers have a single chat scope. */
	get processing() {
		return chatProcessingStore;
	}

	/**
	 * Abort the current agentic flow signal without clearing loading state.
	 * Used by "Send immediately" to force the agentic loop to exit so that
	 * the pending steering message can be re-sent.
	 *
	 * Any tool calls captured mid-stream are dropped before the abort so the
	 * pending message (or a manual follow-up) does not re-send a half-received
	 * tool call with invalid JSON arguments to the server. Mirrors what the
	 * Stop button already does through stopGenerationForChat.
	 */
	async abortCurrentFlow(convId: string): Promise<void> {
		await this.savePartialResponseIfNeeded(convId);
		const c = this.abortControllers.get(convId);

		if (c) {
			c.abort();
			this.abortControllers.delete(convId);
		}
	}

	async addMessage(
		role: MessageRole,
		content: string,
		type: MessageType = MessageType.TEXT,
		parent: string = '-1',
		extras?: DatabaseMessageExtra[],
		isSynthetic?: boolean
	): Promise<DatabaseMessage> {
		const activeConv = conversationsStore.activeConversation;

		if (!activeConv) throw new Error('No active conversation');

		let parentId: string | null = null;

		if (parent === '-1') {
			const am = conversationsStore.activeMessages;

			if (am.length > 0) parentId = am[am.length - 1].id;
			else {
				const all = await conversationsStore.getConversationMessages(activeConv.id);
				const r = all.find((m) => m.parent === null && m.type === 'root');

				parentId = r ? r.id : await DatabaseService.createRootMessage(activeConv.id);
			}
		} else parentId = parent;

		const message = await DatabaseService.createMessageBranch(
			{
				children: [],
				content,
				convId: activeConv.id,
				extra: extras,
				isSynthetic,
				role,
				timestamp: Date.now(),
				toolCalls: '',
				type
			},
			parentId
		);

		conversationsStore.addMessageToActive(message);
		await conversationsStore.updateCurrentNode(message.id);
		conversationsStore.updateConversationTimestamp();

		return message;
	}
	async addSystemPrompt(draft?: { message: string; files: ChatUploadedFile[] }): Promise<void> {
		let activeConv = conversationsStore.activeConversation;

		if (!activeConv) {
			await conversationsStore.createConversation(undefined, draft);
			activeConv = conversationsStore.activeConversation;
		}

		if (!activeConv) return;

		try {
			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);
			const rootId = rootMessage
				? rootMessage.id
				: await DatabaseService.createRootMessage(activeConv.id);
			const existingSystemMessage = allMessages.find(
				(m) => m.role === MessageRole.SYSTEM && m.parent === rootId
			);

			if (existingSystemMessage) {
				this.pendingEditMessageId = existingSystemMessage.id;

				if (!conversationsStore.activeMessages.some((m) => m.id === existingSystemMessage.id))
					conversationsStore.activeMessages.unshift(existingSystemMessage);

				return;
			}

			const am = conversationsStore.activeMessages;
			const firstActiveMessage = am.find((m) => m.parent === rootId);
			const systemMessage = await DatabaseService.createSystemMessage(
				activeConv.id,
				SYSTEM_MESSAGE_PLACEHOLDER,
				rootId
			);

			if (firstActiveMessage) {
				await DatabaseService.updateMessage(firstActiveMessage.id, {
					parent: systemMessage.id
				});
				await DatabaseService.updateMessage(systemMessage.id, {
					children: [firstActiveMessage.id]
				});
				const updatedRootChildren = rootMessage
					? rootMessage.children.filter((id: string) => id !== firstActiveMessage.id)
					: [];

				await DatabaseService.updateMessage(rootId, {
					children: [
						...updatedRootChildren.filter((id: string) => id !== systemMessage.id),
						systemMessage.id
					]
				});
				const firstMsgIndex = conversationsStore.findMessageIndex(firstActiveMessage.id);

				if (firstMsgIndex !== -1)
					conversationsStore.updateMessageAtIndex(firstMsgIndex, {
						parent: systemMessage.id
					});
			}

			conversationsStore.activeMessages.unshift(systemMessage);
			this.pendingEditMessageId = systemMessage.id;
			conversationsStore.updateConversationTimestamp();
		} catch (error) {
			console.error('Failed to add system prompt:', error);
		}
	}
	cancelPreEncode(): void {
		if (this.preEncodeAbortController) {
			this.preEncodeAbortController.abort();
			this.preEncodeAbortController = null;
		}
	}

	/**
	 * Whether the branch holds a step to summarize before the steps a
	 * compaction preserves.
	 */
	canCompact(messages: DatabaseMessage[]): boolean {
		return canCompactMessages(messages, this.compactionPreserve());
	}

	/**
	 * Resets the loading, streaming and processing state for a conversation
	 * after a generation ends or errors. Shared by the flows' exit paths.
	 */
	cleanupStreaming(convId: string): void {
		this.setChatLoading(convId, false);
		this.clearChatStreaming(convId);
		this.processing.setState(convId, null);
	}
	clearChatStreaming(convId: string, messageId?: string): void {
		// session aware: a stale generation must not wipe a newer one's streaming state on the
		// same conversation, that would drop the frozen stop identity and stop the wrong session
		if (messageId !== undefined) {
			const cur = this.chatStreamingStates.get(convId);

			if (cur && cur.messageId !== messageId) return;
		}

		this.chatStreamingStates.delete(convId);
	}
	clearEditMode(): void {
		this.isEditModeActive = false;
		this.addFilesHandler = null;
	}

	clearPendingEditMessageId(): void {
		this.pendingEditMessageId = null;
	}

	clearPendingMessage(convId: string): void {
		this.pendingMessages.delete(convId);
	}

	/**
	 * Compacts the active branch: the model summarizes every turn since the
	 * last compaction boundary and the summary is stored in a new compaction
	 * node appended to the branch. Later requests slice the branch at that
	 * node, so the full history stays visible and navigable while the model
	 * context restarts from the summary.
	 */
	async compactConversation(): Promise<void> {
		const activeConv = conversationsStore.activeConversation;

		if (!activeConv) return;

		const convId = activeConv.id;

		if (this.isChatLoadingInternal(convId) || agenticStore.isRunning(convId)) return;

		// Snapshot of the reactive branch: the compaction node created below
		// must not leak into the summarization request.
		const source = sliceAtLastCompaction([...conversationsStore.activeMessages]);
		const preserve = this.compactionPreserve();
		const cut = compactionCut(source, preserve);

		if (cut === null) return;

		this.showErrorDialog(null);
		this.setChatLoading(convId, true);

		const node = await this.addMessage(
			MessageRole.USER,
			'',
			MessageType.COMPACTION,
			'-1',
			undefined,
			true
		);

		// An interrupted compaction leaves no summary behind, so the node goes
		// along with the request instead of lingering empty in the branch.
		await this.summarizeIntoNode(convId, node, source, async () => {
			const idx = conversationsStore.findMessageIndex(node.id);

			if (idx >= 0) conversationsStore.removeMessageAtIndex(idx);

			await DatabaseService.deleteMessage(node.id);

			if (node.parent) await conversationsStore.updateCurrentNode(node.parent);
		});

		const pending = this.consumePendingMessage(convId);

		if (pending) await this.sendQueuedMessage(convId, pending);
	}

	consumePendingMessage(
		convId: string
	): { content: string; extras?: DatabaseMessageExtra[] } | null {
		const msg = this.pendingMessages.get(convId);

		if (!msg) return null;

		this.pendingMessages.delete(convId);

		return msg;
	}

	async continueAssistantMessage(messageId: string): Promise<void> {
		return this.flows.continueAssistantMessage(messageId);
	}

	async createAssistantMessage(parentId?: string): Promise<DatabaseMessage> {
		const activeConv = conversationsStore.activeConversation;

		if (!activeConv) throw new Error('No active conversation');

		return await DatabaseService.createMessageBranch(
			{
				children: [],
				content: '',
				convId: activeConv.id,
				model: null,
				role: MessageRole.ASSISTANT,
				timestamp: Date.now(),
				toolCalls: '',
				type: MessageType.TEXT
			},
			parentId || null
		);
	}

	async deleteMessage(messageId: string): Promise<void> {
		return this.flows.deleteMessage(messageId);
	}

	/**
	 * Server-side stream sessions (discovery, attach/replay, resume retry,
	 * remote-running snapshot) live in ChatStreamManager.
	 */
	async discoverActiveStream(convId: string): Promise<void> {
		return this.streams.discoverActiveStream(convId);
	}

	dismissErrorDialog(): void {
		this.errorDialogState = null;
	}

	async editAssistantMessage(
		messageId: string,
		newContent: string,
		shouldBranch: boolean
	): Promise<void> {
		return this.flows.editAssistantMessage(messageId, newContent, shouldBranch);
	}

	async editMessageWithBranching(
		messageId: string,
		newContent: string,
		newExtras?: DatabaseMessageExtra[]
	): Promise<void> {
		return this.flows.editMessageWithBranching(messageId, newContent, newExtras);
	}

	async editUserMessagePreserveResponses(
		messageId: string,
		newContent: string,
		newExtras?: DatabaseMessageExtra[]
	): Promise<void> {
		return this.flows.editUserMessagePreserveResponses(messageId, newContent, newExtras);
	}

	/**
	 * Send the message left queued for `convId` once that conversation is
	 * active and idle (e.g. after navigating back to it).
	 */
	async flushPendingMessage(convId: string): Promise<void> {
		if (conversationsStore.activeConversation?.id !== convId) return;

		if (this.isChatLoadingInternal(convId) || agenticStore.isRunning(convId)) return;

		const pending = this.consumePendingMessage(convId);

		if (pending) await this.sendMessage(pending.content, pending.extras);
	}

	getAddFilesHandler(): ((files: File[]) => void) | null {
		return this.addFilesHandler;
	}

	/** Convs with any activity (local pipe or remote session), sidebar spinners. */
	getAllLoadingChats(): string[] {
		return this.activity.loadingConvs;
	}

	getApiOptions(): Record<string, unknown> {
		const currentConfig = settingsStore.config;
		const hasValue = (value: unknown): boolean =>
			value !== undefined && value !== null && value !== '';
		const apiOptions: Record<string, unknown> = { stream: true, timings_per_token: true };

		if (serverStore.isRouterMode) {
			const modelName = modelsStore.selectedModelName;

			if (modelName) apiOptions.model = modelName;
		}

		if (currentConfig.systemMessage) apiOptions.systemMessage = currentConfig.systemMessage;

		if (currentConfig.disableReasoningParsing) apiOptions.disableReasoningParsing = true;

		if (currentConfig.excludeReasoningFromContext) apiOptions.excludeReasoningFromContext = true;

		// a chat bar left empty falls back to the configured prefill
		const reasoningPrefill = this.turnPrefill?.reasoning || currentConfig.reasoningPrefill;
		const responsePrefill = this.turnPrefill?.response || currentConfig.responsePrefill;

		if (reasoningPrefill) apiOptions.reasoningPrefill = reasoningPrefill;

		if (responsePrefill) apiOptions.responsePrefill = responsePrefill;

		// an explicit reasoning choice overrides the server default, DEFAULT sends nothing:
		// a level the chat template honors goes as is, the others as a token budget
		const modelId = modelsStore.selectedModelName ?? '';
		const effort = modelsStore.props.resolveReasoningEffort(
			modelId,
			conversationsStore.preferences.getReasoningEffort()
		);
		const budgetTokens = REASONING_EFFORT_TOKENS[effort];

		if (effort !== ReasoningEffort.DEFAULT) {
			apiOptions.enableThinking = effort !== ReasoningEffort.OFF;
		}

		if (modelsStore.props.getReasoningEfforts(modelId).includes(effort)) {
			apiOptions.reasoningEffort = effort;
		} else if (budgetTokens >= 0) {
			apiOptions.thinkingBudgetTokens = budgetTokens;
		}

		if (hasValue(currentConfig.temperature))
			apiOptions.temperature = Number(currentConfig.temperature);

		if (hasValue(currentConfig.max_tokens))
			apiOptions.max_tokens = Number(currentConfig.max_tokens);

		if (hasValue(currentConfig.dynatemp_range))
			apiOptions.dynatemp_range = Number(currentConfig.dynatemp_range);

		if (hasValue(currentConfig.dynatemp_exponent))
			apiOptions.dynatemp_exponent = Number(currentConfig.dynatemp_exponent);

		if (hasValue(currentConfig.top_k)) apiOptions.top_k = Number(currentConfig.top_k);

		if (hasValue(currentConfig.top_p)) apiOptions.top_p = Number(currentConfig.top_p);

		if (hasValue(currentConfig.min_p)) apiOptions.min_p = Number(currentConfig.min_p);

		if (hasValue(currentConfig.xtc_probability))
			apiOptions.xtc_probability = Number(currentConfig.xtc_probability);

		if (hasValue(currentConfig.xtc_threshold))
			apiOptions.xtc_threshold = Number(currentConfig.xtc_threshold);

		if (hasValue(currentConfig.typ_p)) apiOptions.typ_p = Number(currentConfig.typ_p);

		if (hasValue(currentConfig.repeat_last_n))
			apiOptions.repeat_last_n = Number(currentConfig.repeat_last_n);

		if (hasValue(currentConfig.repeat_penalty))
			apiOptions.repeat_penalty = Number(currentConfig.repeat_penalty);

		if (hasValue(currentConfig.presence_penalty))
			apiOptions.presence_penalty = Number(currentConfig.presence_penalty);

		if (hasValue(currentConfig.frequency_penalty))
			apiOptions.frequency_penalty = Number(currentConfig.frequency_penalty);

		if (hasValue(currentConfig.dry_multiplier))
			apiOptions.dry_multiplier = Number(currentConfig.dry_multiplier);

		if (hasValue(currentConfig.dry_base)) apiOptions.dry_base = Number(currentConfig.dry_base);

		if (hasValue(currentConfig.dry_allowed_length))
			apiOptions.dry_allowed_length = Number(currentConfig.dry_allowed_length);

		if (hasValue(currentConfig.dry_penalty_last_n))
			apiOptions.dry_penalty_last_n = Number(currentConfig.dry_penalty_last_n);

		if (currentConfig.samplers) apiOptions.samplers = currentConfig.samplers;

		if (hasValue(currentConfig.backend_sampling))
			apiOptions.backend_sampling = currentConfig.backend_sampling;

		if (currentConfig.customJson) apiOptions.custom = currentConfig.customJson;

		return apiOptions;
	}

	getChatStreaming(convId: string): { response: string; messageId: string } | undefined {
		return this.getChatStreamingState(convId);
	}

	/** Node id of the summary running for the conv, or null when none runs. */
	getCompactingNodeId(convId: string): string | null {
		return this.activity.getCompactingNodeId(convId);
	}

	async getDeletionInfo(messageId: string): Promise<{
		totalCount: number;
		userMessages: number;
		assistantMessages: number;
		messageTypes: string[];
	}> {
		return this.flows.getDeletionInfo(messageId);
	}

	getOrCreateAbortController(convId: string): AbortController {
		let c = this.abortControllers.get(convId);

		if (!c || c.signal.aborted) {
			c = new AbortController();
			this.abortControllers.set(convId, c);
		}

		return c;
	}

	getPendingMessageContent(convId: string): string | null {
		return this.pendingMessages.get(convId)?.content ?? null;
	}

	getPendingMessageExtras(convId: string): DatabaseMessageExtra[] | undefined {
		return this.pendingMessages.get(convId)?.extras;
	}

	getResumeModel(convId: string): string | null {
		return this.streams.getResumeModel(convId);
	}

	hasPendingMessage(convId: string): boolean {
		return this.pendingMessages.has(convId);
	}

	injectPendingMessage(convId: string, content: string, extras?: DatabaseMessageExtra[]): void {
		this.pendingMessages.set(convId, { content, extras });
	}

	isChatLoading(convId: string): boolean {
		return this.activity.isLocal(convId);
	}

	isChatLoadingInternal(convId: string): boolean {
		return this.activity.isLocal(convId) || this.chatStreamingStates.has(convId);
	}

	/**
	 * True while the conv's in-flight run is a compaction summary, which reports
	 * its own progress on the compaction row.
	 */
	isCompacting(convId: string): boolean {
		return this.activity.isCompacting(convId);
	}

	isEditing(): boolean {
		return this.isEditModeActive;
	}

	/** True while the active conversation has a live streaming pipe. */
	isStreaming(): boolean {
		return this.chatStreamingStates.has(conversationsStore.activeConversation?.id ?? '');
	}

	/**
	 * Record a working-directory change into chat history as a synthetic
	 * user message, so the model sees it on its next turn (the client
	 * sends the cwd itself via the x-tool-cwd header on tool calls).
	 * A plain user message is used because some chat templates reject
	 * tool messages without a preceding tool call.
	 */
	async recordCwdChange(cwd: string | null): Promise<void> {
		const content = cwd
			? formatCwdMessage(cwd, await toolsStore.resolveServerHome())
			: CWD_CLEARED_TEXT;
		// Reuse the trailing cwd row when it is already the last message, so
		// repeated picks update it in place instead of stacking another row.
		const last = conversationsStore.activeMessages[conversationsStore.activeMessages.length - 1];

		if (last && last.role === MessageRole.USER && last.isSynthetic === true) {
			await DatabaseService.updateMessage(last.id, { content, isSynthetic: true });
			conversationsStore.updateMessageAtIndex(conversationsStore.activeMessages.length - 1, {
				content,
				isSynthetic: true
			});

			return;
		}

		await this.addMessage(MessageRole.USER, content, MessageType.TEXT, '-1', undefined, true);
	}

	/**
	 * Re-runs the summary of an existing compaction node in place. The node
	 * keeps its position in the branch, so the turns below it stay untouched
	 * while the summarized context is rewritten.
	 */
	async regenerateCompaction(messageId: string): Promise<void> {
		const activeConv = conversationsStore.activeConversation;

		if (!activeConv) return;

		const convId = activeConv.id;

		if (this.isChatLoadingInternal(convId) || agenticStore.isRunning(convId)) return;

		const allMessages = await conversationsStore.getConversationMessages(convId);
		const node = findMessageById(allMessages, messageId);

		if (!node || node.type !== MessageType.COMPACTION) return;

		// The summary covers the branch above the node: walking the path up to
		// its parent keeps the old summary from being folded into the new one.
		const branchPath = node.parent ? [...filterByLeafNodeId(allMessages, node.parent, false)] : [];
		const source = sliceAtLastCompaction(branchPath);

		if (!this.canCompact(source)) return;

		this.showErrorDialog(null);
		this.setChatLoading(convId, true);

		const previous = { content: node.content, model: node.model };

		await this.summarizeIntoNode(convId, node, source, async () => {
			const idx = conversationsStore.findMessageIndex(node.id);

			if (idx >= 0) conversationsStore.updateMessageAtIndex(idx, previous);

			await DatabaseService.updateMessage(node.id, previous);
		});
	}

	async regenerateMessage(messageId: string): Promise<void> {
		return this.flows.regenerateMessage(messageId);
	}

	async regenerateMessageWithBranching(messageId: string, modelOverride?: string): Promise<void> {
		return this.flows.regenerateMessageWithBranching(messageId, modelOverride);
	}

	async removeSystemPromptPlaceholder(messageId: string): Promise<boolean> {
		const activeConv = conversationsStore.activeConversation;

		if (!activeConv) return false;

		try {
			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const systemMessage = findMessageById(allMessages, messageId);

			if (!systemMessage || systemMessage.role !== MessageRole.SYSTEM) return false;

			const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);

			if (!rootMessage) return false;

			if (allMessages.length === 2 && systemMessage.children.length === 0) {
				await conversationsStore.deleteConversation(activeConv.id);

				return true;
			}

			for (const childId of systemMessage.children) {
				await DatabaseService.updateMessage(childId, { parent: rootMessage.id });
				const childIndex = conversationsStore.findMessageIndex(childId);

				if (childIndex !== -1)
					conversationsStore.updateMessageAtIndex(childIndex, { parent: rootMessage.id });
			}
			await DatabaseService.updateMessage(rootMessage.id, {
				children: [
					...rootMessage.children.filter((id: string) => id !== messageId),
					...systemMessage.children
				]
			});
			await DatabaseService.deleteMessage(messageId);
			const systemIndex = conversationsStore.findMessageIndex(messageId);

			if (systemIndex !== -1) conversationsStore.activeMessages.splice(systemIndex, 1);

			conversationsStore.updateConversationTimestamp();

			return false;
		} catch (error) {
			console.error('Failed to remove system prompt placeholder:', error);

			return false;
		}
	}

	/**
	 * Hydra send: the conversation so far plus the input box (if any) is the root, each tail in
	 * hydraStore is one head. Every head is stored as a branch under the root: its tail as a user
	 * message, its reply as the child. The chat then shows head 0's branch; the hydra group view
	 * shows all heads side by side.
	 */
	async sendHydra(content: string, extras?: DatabaseMessageExtra[]): Promise<void> {
		const tails = hydraStore.tails.map((t) => t.trim());

		let activeConv = conversationsStore.activeConversation;

		if (activeConv && this.isChatLoadingInternal(activeConv.id)) {
			toast.error('Wait for the running reply to finish before sending a hydra');

			return;
		}

		const hasRootInput = content.trim().length > 0 || (extras?.length ?? 0) > 0;

		if (!hasRootInput && !conversationsStore.activeMessages.some((m) => m.type !== 'root')) {
			toast.error('A hydra needs a root: type a message, or send it in a chat that has messages');

			return;
		}

		this.cancelPreEncode();

		let isNewConversation = false;

		if (!activeConv) {
			await conversationsStore.createConversation();
			activeConv = conversationsStore.activeConversation;
			isNewConversation = true;
		}

		if (!activeConv) return;

		const convId = activeConv.id;

		this.showErrorDialog(null);
		this.setChatLoading(convId, true);

		const abortController = this.getOrCreateAbortController(convId);
		// the reply each head writes into now; a loop guard note starts a new one
		const current: string[] = [];
		const markers: DatabaseMessageHydra[] = [];
		// every reply of this send that is not saved yet
		const unsaved = new SvelteSet<string>();
		const newMessage = (
			role: MessageRole,
			content: string,
			hydra: DatabaseMessageHydra,
			parent: string
		) =>
			DatabaseService.createMessageBranch(
				{
					children: [],
					content,
					convId,
					hydra,
					model: role === MessageRole.ASSISTANT ? null : undefined,
					role,
					timestamp: Date.now(),
					toolCalls: '',
					type: MessageType.TEXT
				},
				parent
			);
		const saveReply = async (id: string) => {
			const live = hydraStore.live.get(id);

			unsaved.delete(id);

			if (!live) return;

			const update: Partial<DatabaseMessage> = {
				content: live.content,
				reasoningContent: live.reasoning || undefined,
				timings: live.timings
			};

			await DatabaseService.updateMessage(id, update);

			const idx = conversationsStore.findMessageIndex(id);

			if (idx !== -1) conversationsStore.updateMessageAtIndex(idx, update);

			hydraStore.live.delete(id);
		};

		try {
			let rootParentId = isNewConversation ? await this.startNewConversation(activeConv) : '-1';

			if (hasRootInput) {
				const userMessage = await this.addMessage(
					MessageRole.USER,
					content,
					MessageType.TEXT,
					rootParentId,
					extras
				);

				rootParentId = userMessage.id;

				if (isNewConversation && content)
					await conversationsStore.applyTitleFromContent(convId, content);
			} else if (rootParentId === '-1') {
				rootParentId =
					conversationsStore.activeMessages[conversationsStore.activeMessages.length - 1].id;
			}

			// the root as the server sees it: the active path, up to the root's last message
			const rootMessages = [...conversationsStore.activeMessages];
			const group = uuid();

			for (let i = 0; i < tails.length; i++) {
				const hydra = { count: tails.length, group, index: i };
				const tail = await newMessage(MessageRole.USER, tails[i], hydra, rootParentId);
				const reply = await newMessage(MessageRole.ASSISTANT, '', hydra, tail.id);

				current.push(reply.id);
				markers.push(hydra);
				unsaved.add(reply.id);
				hydraStore.updateLive(reply.id, { content: '', done: false, reasoning: '' });
			}

			// show head 0's branch, the group view lists the others next to it
			await conversationsStore.updateCurrentNode(current[0]);
			await conversationsStore.refreshActiveMessages();
			// keep the tails, so the same heads can go to the next root; hide the boxes to free the screen
			hydraStore.collapsed = true;

			await ChatService.sendHydra(
				rootMessages,
				tails,
				hydraStore.thinking === 'default'
					? this.getApiOptions()
					: { ...this.getApiOptions(), enableThinking: hydraStore.thinking === 'on' },
				{
					onHeadChunk: (index, delta) => {
						const id = current[index];
						const cur = id ? hydraStore.live.get(id) : undefined;

						if (!id || !cur) return;

						hydraStore.updateLive(id, {
							content: cur.content + (delta.content ?? ''),
							reasoning: cur.reasoning + (delta.reasoning ?? ''),
							reflecting: false
						});
					},
					onHeadDone: (index, finishReason, timings) => {
						const id = current[index];

						if (id) hydraStore.updateLive(id, { done: true, finishReason, timings });
					},
					// llama-server --loop-guard on a head: the reflection shows in the head's column, a note
					// ends the reply so far and starts a new one in the head's branch, as in a normal chat
					onHeadLoopGuard: async (index, event) => {
						const id = current[index];
						const cur = id ? hydraStore.live.get(id) : undefined;

						if (!id || !cur) return;

						if (event.event === 'reflect_start') {
							hydraStore.updateLive(id, { reflecting: true, reflection: '' });
						} else if (event.event === 'reflect_delta') {
							hydraStore.updateLive(id, {
								reflection: (cur.reflection ?? '') + (event.text ?? '')
							});
						} else if (event.event === 'resume') {
							hydraStore.updateLive(id, { reflecting: false });
						} else if (event.event === 'note' && event.message) {
							hydraStore.updateLive(id, {
								done: true,
								finishReason: 'loop guard',
								reflecting: false
							});
							await saveReply(id);

							const note = await newMessage(
								MessageRole.SYSTEM,
								event.message.content,
								markers[index],
								id
							);
							const next = await newMessage(MessageRole.ASSISTANT, '', markers[index], note.id);

							current[index] = next.id;
							unsaved.add(next.id);
							hydraStore.updateLive(next.id, { content: '', done: false, reasoning: '' });

							// the chat shows this head: keep its path at the head's newest reply
							if (conversationsStore.activeConversation?.currNode === id) {
								await conversationsStore.updateCurrentNode(next.id);
								await conversationsStore.refreshActiveMessages();
							}
						}
					},
					onHeadProgress: (index, progress) => {
						const id = current[index];

						if (id) hydraStore.updateLive(id, { progress });
					},
					onRootProgress: (progress) => hydraStore.rootProgress.set(convId, progress)
				},
				abortController.signal
			);
		} catch (error) {
			if (!isAbortError(error)) {
				console.error('Hydra request failed:', error);
				this.showErrorDialog({
					message: error instanceof Error ? error.message : 'Unknown error',
					type: ErrorDialogType.SERVER
				});
			}
		} finally {
			// keep what each head wrote, also after a stop or an error
			for (const id of [...unsaved]) await saveReply(id);

			hydraStore.rootProgress.delete(convId);
			this.abortControllers.delete(convId);
			this.setChatLoading(convId, false);
			conversationsStore.updateConversationTimestamp();
		}
	}

	async sendMessage(
		content: string,
		extras?: DatabaseMessageExtra[],
		prefill?: { reasoning: string; response: string }
	): Promise<void> {
		if (hydraStore.ready) return this.sendHydra(content, extras);

		if (!content.trim() && (!extras || extras.length === 0)) return;

		const activeConv = conversationsStore.activeConversation;

		// If agentic loop is running, inject as a steering message instead of starting a new flow
		if (activeConv && agenticStore.isRunning(activeConv.id)) {
			agenticStore.injectSteeringMessage(activeConv.id, content, extras);

			return;
		}

		// If non-agentic streaming is active, queue as a pending message to send after completion
		if (activeConv && this.isChatLoadingInternal(activeConv.id)) {
			this.injectPendingMessage(activeConv.id, content, extras);

			return;
		}

		// Cancel any in-flight pre-encode request
		this.cancelPreEncode();

		// Consume MCP resource attachments - converts them to extras and clears the live store
		const resourceExtras = mcpStore.consumeResourceAttachmentsAsExtras();
		const allExtras = resourceExtras.length > 0 ? [...(extras || []), ...resourceExtras] : extras;

		let isNewConversation = false;

		if (!activeConv) {
			await conversationsStore.createConversation();
			isNewConversation = true;
		}

		const currentConv = conversationsStore.activeConversation;

		if (!currentConv) return;

		this.showErrorDialog(null);
		this.setChatLoading(currentConv.id, true);
		this.clearChatStreaming(currentConv.id);
		try {
			const parentIdForUserMessage = isNewConversation
				? await this.startNewConversation(currentConv)
				: undefined;
			const userMessage = await this.addMessage(
				MessageRole.USER,
				content,
				MessageType.TEXT,
				parentIdForUserMessage ?? '-1',
				allExtras
			);

			if (isNewConversation && content)
				await conversationsStore.applyTitleFromContent(currentConv.id, content);

			const assistantMessage = await this.createAssistantMessage(userMessage.id);

			conversationsStore.addMessageToActive(assistantMessage);
			// this turn's prefill only, read by getApiOptions while the request is built
			this.turnPrefill = prefill ?? null;

			try {
				await this.streamChatCompletion(
					conversationsStore.activeMessages.slice(0, -1),
					assistantMessage,
					undefined,
					undefined,
					undefined,
					settingsStore.config.titleGenerationUseLLM && isNewConversation ? content : undefined
				);
			} finally {
				this.turnPrefill = null;
			}
		} catch (error) {
			if (isAbortError(error)) {
				this.setChatLoading(currentConv.id, false);

				return;
			}

			console.error('Failed to send message:', error);
			this.setChatLoading(currentConv.id, false);
			const dialogType =
				error instanceof Error && error.name === 'TimeoutError'
					? ErrorDialogType.TIMEOUT
					: ErrorDialogType.SERVER;
			const contextInfo = (
				error as Error & { contextInfo?: { n_prompt_tokens: number; n_ctx: number } }
			).contextInfo;

			this.showErrorDialog({
				contextInfo,
				message: error instanceof Error ? error.message : 'Unknown error',
				type: dialogType
			});
		}
	}
	setChatLoading(convId: string, loading: boolean): void {
		if (loading) {
			this.activity.markLocal(convId);
		} else {
			this.activity.localEnded(convId);
			this.setChatReasoning(convId, false);
		}
	}

	setChatReasoning(convId: string, reasoning: boolean): void {
		if (reasoning) this.chatReasoningStates.set(convId, true);
		else this.chatReasoningStates.delete(convId);
	}

	setChatStreaming(
		convId: string,
		response: string,
		messageId: string,
		model?: string | null
	): void {
		this.chatStreamingStates.set(convId, {
			messageId,
			model: model ?? this.chatStreamingStates.get(convId)?.model,
			response
		});
	}

	setEditModeActive(handler: (files: File[]) => void): void {
		this.isEditModeActive = true;
		this.addFilesHandler = handler;
	}

	showErrorDialog(state: ErrorDialogState | null): void {
		this.errorDialogState = state;
	}

	async stopGeneration(): Promise<void> {
		const activeConv = conversationsStore.activeConversation;

		if (!activeConv) return;

		await this.stopGenerationForChat(activeConv.id);
	}

	async stopGenerationForChat(convId: string): Promise<void> {
		await this.savePartialResponseIfNeeded(convId);
		// tell the server to stop the generation, not just drop the HTTP socket. without this the
		// detached drain keeps producing tokens until eos or max_tokens. use the frozen identity
		// captured when the session started, not the live dropdown
		const streamStateForStop = this.chatStreamingStates.get(convId);
		const modelForStop = streamStateForStop?.model ?? ChatService.getStreamState(convId)?.model;

		void ChatService.cancelServerStream(convId, modelForStop);
		// an explicit stop leaves nothing to resume and kills a pending resume retry
		ChatService.clearStreamState(convId);
		this.streams.cancelResumeRetry(convId);
		this.abortRequest(convId);
		this.setChatLoading(convId, false);
		this.clearChatStreaming(convId);
		this.processing.setState(convId, null);
		this.clearPendingMessage(convId);
	}

	async streamChatCompletion(
		allMessages: DatabaseMessage[],
		assistantMessage: DatabaseMessage,
		onComplete?: (content: string) => Promise<void>,
		onError?: (error: Error) => void,
		modelOverride?: string | null,
		firstUserMessageContent?: string
	): Promise<void> {
		// the ::model suffix in the stream identity is only for router mode, where it routes to the
		// owning child. in single-model mode the identity stays the bare conv id so that attach, stop
		// and reattach all agree, regardless of fresh send vs regenerate passing a resolved model
		let effectiveModel: string | null | undefined = undefined;

		if (serverStore.isRouterMode) {
			const conversationModel = getConversationModel(allMessages);

			effectiveModel = modelOverride || modelsStore.selectedModelName || conversationModel;
		}

		// a router serves the props the request options read once the model is loaded, so the
		// request waits for the load, and a stop during the wait aborts it
		const abortController = this.getOrCreateAbortController(assistantMessage.convId);

		if (serverStore.isRouterMode && effectiveModel) {
			await modelsStore.status.ensureLoaded(effectiveModel);

			if (abortController.signal.aborted) return;

			if (!modelsStore.props.getModelProps(effectiveModel))
				await modelsStore.props.fetchModelProps(effectiveModel);
		}

		// Every request path funnels through here, so the compaction boundary
		// applied once covers the plain send, the agentic flow, and the
		// pre-encode that reuses this list from the closure.
		allMessages = sliceAtLastCompaction(allMessages);

		// Mutable state for the current message being streamed
		let currentMessageId = assistantMessage.id;
		let streamedContent = '';
		let streamedReasoningContent = '';
		let resolvedModel: string | null = null;
		let modelPersisted = false;

		const convId = assistantMessage.convId;

		// Tracks the last message created in this flow. Used as the parent for the next
		// turn's assistant message so createAssistantMessage does not have to read
		// conversationsStore.activeMessages, which may belong to a different conversation
		// after the user navigates while the loop is still running.
		let lastCreatedInFlow = currentMessageId;

		// freeze the POST identity from t0 so a stop cancels with the exact session key,
		// never a stale or empty model resolved later
		this.setChatStreaming(convId, streamedContent, currentMessageId, effectiveModel);

		const recordModel = (modelName: string | null | undefined, persistImmediately = true): void => {
			if (!modelName) return;

			const n = normalizeModelName(modelName);

			if (!n || n === resolvedModel) return;

			resolvedModel = n;
			const idx = conversationsStore.findMessageIndex(currentMessageId);

			conversationsStore.updateMessageAtIndex(idx, { model: n });

			if (persistImmediately && !modelPersisted) {
				modelPersisted = true;
				DatabaseService.updateMessage(currentMessageId, { model: n }).catch(() => {
					modelPersisted = false;
					resolvedModel = null;
				});
			}
		};

		let completionIdRecorded = false;

		const recordCompletionId = (id: string): void => {
			if (!id || completionIdRecorded) return;

			completionIdRecorded = true;
			const idx = conversationsStore.findMessageIndex(currentMessageId);

			conversationsStore.updateMessageAtIndex(idx, { completionId: id });
			DatabaseService.updateMessage(currentMessageId, { completionId: id }).catch(() => {
				completionIdRecorded = false;
			});
		};
		const updateStreamingUI = () => {
			this.setChatStreaming(convId, streamedContent, currentMessageId, effectiveModel);
			const idx = conversationsStore.findMessageIndex(currentMessageId);

			conversationsStore.updateMessageAtIndex(idx, { content: streamedContent });
		};
		const cleanupStreamingState = () => {
			this.setChatLoading(convId, false);
			this.clearChatStreaming(convId, currentMessageId);
			this.processing.setState(convId, null);
		};

		this.processing.setActiveConversation(convId);
		let hat: HatWear | null = null;
		// llama-server --loop-guard split the reply: this flow made more than one message
		let loopGuardSplit = false;
		let loopGuardReflection = '';

		const streamCallbacks: ChatStreamCallbacks = {
			createAssistantMessage: async () => {
				// Reset streaming state for new message
				streamedContent = '';
				streamedReasoningContent = '';

				const msg = await DatabaseService.createMessageBranch(
					{
						children: [],
						content: '',
						convId,
						model: resolvedModel,
						role: MessageRole.ASSISTANT,
						timestamp: Date.now(),
						toolCalls: '',
						type: MessageType.TEXT
					},
					lastCreatedInFlow
				);

				if (conversationsStore.activeConversation?.id === convId) {
					conversationsStore.addMessageToActive(msg);
				}

				currentMessageId = msg.id;
				lastCreatedInFlow = msg.id;

				return msg;
			},
			createLoopGuardNote: async (content: string) => {
				// a real message of the conversation: the server holds it in its cache at this
				// place, so it has to be sent back at this place. type TEXT keeps it visible
				const msg = await DatabaseService.createMessageBranch(
					{
						children: [],
						content,
						convId,
						role: MessageRole.SYSTEM,
						timestamp: Date.now(),
						toolCalls: '',
						type: MessageType.TEXT
					},
					currentMessageId
				);

				if (conversationsStore.activeConversation?.id === convId) {
					conversationsStore.addMessageToActive(msg);
				}

				lastCreatedInFlow = msg.id;
				loopGuardSplit = true;

				return msg;
			},
			createToolResultMessage: async (
				toolCallId: string,
				content: string,
				extras?: DatabaseMessageExtra[],
				toolCwd?: string
			) => {
				const msg = await DatabaseService.createMessageBranch(
					{
						children: [],
						content,
						convId,
						extra: extras,
						role: MessageRole.TOOL,
						timestamp: Date.now(),
						toolCallId,
						toolCalls: '',
						toolCwd,
						type: MessageType.TEXT
					},
					currentMessageId
				);

				// mirror into the active store and move the node pointer only when this
				// conversation is displayed; otherwise persist the node move straight to
				// the db for the owning conv so a foreign conv's currNode stays untouched
				if (conversationsStore.activeConversation?.id === convId) {
					conversationsStore.addMessageToActive(msg);
					await conversationsStore.updateCurrentNode(msg.id);
				} else {
					await DatabaseService.updateCurrentNode(convId, msg.id);
				}

				lastCreatedInFlow = msg.id;

				return msg;
			},
			onAssistantTurnComplete: async (
				content: string,
				reasoningContent: string | undefined,
				timings: ChatMessageTimings | undefined,
				toolCalls: import('$lib/types/api').ApiChatCompletionToolCall[] | undefined
			) => {
				const updateData: Record<string, unknown> = {
					content,
					reasoningContent: reasoningContent || undefined,
					timings,
					toolCalls: toolCalls ? JSON.stringify(toolCalls) : ''
				};

				if (resolvedModel && !modelPersisted) updateData.model = resolvedModel;

				await DatabaseService.updateMessage(currentMessageId, updateData);
				const idx = conversationsStore.findMessageIndex(currentMessageId);
				const uiUpdate: Partial<DatabaseMessage> = {
					content,
					reasoningContent: reasoningContent || undefined,
					toolCalls: toolCalls ? JSON.stringify(toolCalls) : ''
				};

				if (timings) uiUpdate.timings = timings;

				if (resolvedModel) uiUpdate.model = resolvedModel;

				// touch the active ui array and node pointer only when this conversation
				// is displayed; otherwise persist the node move straight to the db so a
				// foreign conv's currNode stays untouched
				if (conversationsStore.activeConversation?.id === convId) {
					conversationsStore.updateMessageAtIndex(idx, uiUpdate);
					await conversationsStore.updateCurrentNode(currentMessageId);
				} else {
					await DatabaseService.updateCurrentNode(convId, currentMessageId);
				}
			},
			onAttachments: (messageId: string, extras: DatabaseMessageExtra[]) => {
				if (!extras.length) return;

				const idx = conversationsStore.findMessageIndex(messageId);

				if (idx === -1) return;

				const msg = conversationsStore.activeMessages[idx];
				const updatedExtras = [...(msg.extra || []), ...extras];

				conversationsStore.updateMessageAtIndex(idx, { extra: updatedExtras });
				DatabaseService.updateMessage(messageId, { extra: updatedExtras }).catch(console.error);
			},
			onChunk: (chunk: string) => {
				streamedContent += chunk;
				updateStreamingUI();
				this.setChatReasoning(convId, false);
			},
			onCompletionId: (id: string) => recordCompletionId(id),
			onError: async (error: Error) => {
				if (isAbortError(error)) {
					cleanupStreamingState();
					// If aborted with a pending message (e.g. "Send immediately"), re-send it
					const pending = this.consumePendingMessage(convId);

					if (pending) {
						void this.sendQueuedMessage(convId, pending);
					}

					return;
				}

				console.error('Streaming error:', error);
				// keep whatever was streamed so far, the message stays in memory and in DB
				await this.savePartialResponseIfNeeded(convId);
				cleanupStreamingState();
				this.clearPendingMessage(convId);

				const contextInfo = (
					error as Error & { contextInfo?: { n_prompt_tokens: number; n_ctx: number } }
				).contextInfo;

				this.showErrorDialog({
					contextInfo,
					message: error.message,
					type: error.name === 'TimeoutError' ? ErrorDialogType.TIMEOUT : ErrorDialogType.SERVER
				});

				if (onError) onError(error);
			},
			onFlowComplete: (finalTimings?: ChatMessageTimings) => {
				if (finalTimings) {
					const idx = conversationsStore.findMessageIndex(assistantMessage.id);

					conversationsStore.updateMessageAtIndex(idx, { timings: finalTimings });
					DatabaseService.updateMessage(assistantMessage.id, {
						timings: finalTimings
					}).catch(console.error);
				}

				cleanupStreamingState();

				if (onComplete) onComplete(streamedContent);

				if (serverStore.isRouterMode) modelsStore.fetchRouterModels().catch(console.error);

				// Pre-encode conversation in KV cache for faster next turn.
				// Not with a hat: the pre-encode prompt has no tools, so it would displace the hat's cache.
				// Not after a loop guard split: the pre-encode prompt knows only one assistant message
				if (settingsStore.config.preEncodeConversation && !hat && !loopGuardSplit) {
					void this.triggerPreEncode(convId, lastCreatedInFlow, effectiveModel);
				}
			},
			onLoopGuardReflection: (event: ApiLoopGuardEvent) => {
				// the reflection is the server talking about the reply, it is not part of any message
				if (event.event === 'reflect_start') {
					loopGuardReflection = '';
					toast.info('Loop guard: the reply repeats itself, reflecting...');
				} else if (event.event === 'reflect_delta') {
					loopGuardReflection += event.text ?? '';
				} else if (event.event === 'resume') {
					toast.info('Loop guard: no loop found, the reply goes on', {
						description: loopGuardReflection.trim() || undefined,
						duration: 15000
					});
				}
			},
			onModel: (modelName: string) => recordModel(modelName),
			onReasoningChunk: (chunk: string) => {
				streamedReasoningContent += chunk;
				// mark streaming state so a stop mid-thinking can persist the partial reasoning
				this.setChatStreaming(convId, streamedContent, currentMessageId, effectiveModel);
				const idx = conversationsStore.findMessageIndex(currentMessageId);

				conversationsStore.updateMessageAtIndex(idx, {
					reasoningContent: streamedReasoningContent
				});
				this.setChatReasoning(convId, true);
			},
			onTimings: (timings?: ChatMessageTimings, promptProgress?: ChatMessagePromptProgress) => {
				this.processing.applyStreamTimings(timings, promptProgress, convId);
			},
			onToolCallsStreaming: (toolCalls) => {
				const idx = conversationsStore.findMessageIndex(currentMessageId);

				conversationsStore.updateMessageAtIndex(idx, {
					toolCalls: JSON.stringify(toolCalls)
				});
			},
			onTurnComplete: (intermediateTimings: ChatMessageTimings) => {
				// Update the first assistant message with cumulative agentic timings
				const idx = conversationsStore.findMessageIndex(assistantMessage.id);

				conversationsStore.updateMessageAtIndex(idx, { timings: intermediateTimings });
			},
			updateToolResultMessage: async (
				messageId: string,
				content: string,
				extras?: DatabaseMessageExtra[]
			) => {
				// Persist latest content + merged extras; mirror into the active
				// store so the chat view sees live updates for streaming tools
				// (e.g. exec_shell_command). The existing tool message node
				// pointer stays put - the renderer is already scoped to it.
				const updates: Partial<DatabaseMessage> = { content };

				if (extras) {
					const idx = conversationsStore.findMessageIndex(messageId);
					const existing = idx >= 0 ? (conversationsStore.activeMessages[idx]?.extra ?? []) : [];
					const merged = [...existing, ...extras];

					updates.extra = merged;
				}

				if (conversationsStore.activeConversation?.id === convId) {
					const idx = conversationsStore.findMessageIndex(messageId);

					if (idx >= 0) conversationsStore.updateMessageAtIndex(idx, updates);
				}

				await DatabaseService.updateMessage(messageId, updates);
			}
		};
		const toolPolicy = this.toolPolicy();

		try {
			hat = await this.wearHat(convId, abortController.signal);
		} catch (error) {
			await streamCallbacks.onError?.(error instanceof Error ? error : new Error(String(error)));

			return;
		}

		// with a hat the request starts exactly as the hat does: its system text, its tools
		const requestMessages = hat
			? hatsStore.applyToMessages(hat, await ChatService.normalizeMessagesForApi(allMessages))
			: allMessages;

		{
			const agenticResult = await agenticStore.runAgenticFlow({
				callbacks: streamCallbacks,
				conversationId: convId,
				flowRootMessageId: assistantMessage.id,
				messages: requestMessages,
				options: {
					...this.getApiOptions(),
					...(effectiveModel ? { model: effectiveModel } : {})
				},
				signal: abortController.signal,
				toolPolicy,
				tools: hat?.tools
			});

			if (agenticResult.handled) {
				// Generate LLM based title for new conversations after agentic flow completes
				if (firstUserMessageContent) {
					await this.generateTitleWithLLM(firstUserMessageContent, streamedContent, convId);
				}

				// Check if there's a pending steering message to re-send
				const pending = agenticStore.consumePendingSteeringMessage(convId);

				if (pending) {
					await this.sendQueuedMessage(convId, pending);
				}

				return;
			}
		}

		await ChatService.sendMessage(
			requestMessages,
			{
				...this.getApiOptions(),
				...(effectiveModel ? { model: effectiveModel } : {}),
				onChunk: streamCallbacks.onChunk,
				onComplete: async (
					finalContent?: string,
					reasoningContent?: string,
					timings?: ChatMessageTimings,
					toolCalls?: string
				) => {
					const content = streamedContent || finalContent || '';
					const reasoning = streamedReasoningContent || reasoningContent;
					const updateData: Record<string, unknown> = {
						content,
						reasoningContent: reasoning || undefined,
						timings,
						toolCalls: toolCalls || ''
					};

					if (resolvedModel && !modelPersisted) updateData.model = resolvedModel;

					await DatabaseService.updateMessage(currentMessageId, updateData);
					const idx = conversationsStore.findMessageIndex(currentMessageId);
					const uiUpdate: Partial<DatabaseMessage> = {
						content,
						reasoningContent: reasoning || undefined,
						toolCalls: toolCalls || ''
					};

					if (timings) uiUpdate.timings = timings;

					if (resolvedModel) uiUpdate.model = resolvedModel;

					conversationsStore.updateMessageAtIndex(idx, uiUpdate);
					await conversationsStore.updateCurrentNode(currentMessageId);
					cleanupStreamingState();

					// Pre-encode conversation in KV cache for faster next turn
					if (settingsStore.config.preEncodeConversation) {
						void this.triggerPreEncode(convId, currentMessageId, effectiveModel);
					}

					if (onComplete) await onComplete(content);

					if (serverStore.isRouterMode) modelsStore.fetchRouterModels().catch(console.error);

					// Generate LLM based title for new conversations (avoids stale reference
					// issue when user switches conversations while streaming)
					if (firstUserMessageContent) {
						await this.generateTitleWithLLM(firstUserMessageContent, streamedContent, convId);
					}

					// Check if there's a pending message queued during streaming
					const pending = this.consumePendingMessage(convId);

					if (pending) {
						await this.sendQueuedMessage(convId, pending);
					}
				},
				onCompletionId: streamCallbacks.onCompletionId,
				onConnectionState: (state: StreamConnectionState) => {
					if (convId === conversationsStore.activeConversation?.id) {
						this.streamConnectionState = state;
					}
				},
				onError: streamCallbacks.onError,
				onLoopGuard: async (event: ApiLoopGuardEvent) => {
					if (event.event !== 'note' || !event.message) {
						streamCallbacks.onLoopGuardReflection?.(event);

						return;
					}

					// same as the agentic loop: the reply so far, then the note, then a new reply
					await streamCallbacks.onAssistantTurnComplete?.(
						streamedContent,
						streamedReasoningContent || undefined,
						undefined,
						undefined
					);
					await streamCallbacks.createLoopGuardNote?.(event.message.content);
					await streamCallbacks.createAssistantMessage?.();
				},
				onModel: streamCallbacks.onModel,
				onReasoningChunk: streamCallbacks.onReasoningChunk,
				onTimings: streamCallbacks.onTimings,
				stream: true
			},
			convId,
			abortController.signal
		);
	}

	syncLoadingStateForChat(convId: string): void {
		const s = this.chatStreamingStates.get(convId);

		this.processing.setActiveConversation(convId);

		// Sync streaming content to activeMessages so UI displays current content
		if (s?.response && s?.messageId) {
			const idx = conversationsStore.findMessageIndex(s.messageId);

			if (idx !== -1) {
				conversationsStore.updateMessageAtIndex(idx, { content: s.response });
			}
		}
	}

	async syncRemoteRunningStreams(): Promise<void> {
		return this.streams.syncRemoteRunningStreams();
	}

	/**
	 * Message flows (edit / regenerate / continue / delete) live in
	 * ChatMessageFlows; these delegate so consumers keep a single entry point.
	 */
	async updateMessage(messageId: string, newContent: string): Promise<void> {
		return this.flows.updateMessage(messageId, newContent);
	}

	private abortRequest(convId?: string): void {
		if (convId) {
			const c = this.abortControllers.get(convId);

			if (c) {
				c.abort();
				this.abortControllers.delete(convId);
			}
		} else {
			for (const c of this.abortControllers.values()) c.abort();
			this.abortControllers.clear();
		}
	}

	/** The steps a compaction keeps as written, as the settings set them. */
	private compactionPreserve(): number {
		return Number(settingsStore.config.compactionPreservedSteps);
	}

	private async generateTitleWithLLM(
		userContent: string,
		assistantContent: string,
		convId: string
	): Promise<void> {
		const effectiveModel =
			serverStore.isRouterMode && modelsStore.selectedModelName
				? modelsStore.selectedModelName
				: undefined;
		const configValue = settingsStore.config;
		const titlePromptTemplate =
			typeof configValue.titleGenerationPrompt === 'string' &&
			configValue.titleGenerationPrompt.trim()
				? configValue.titleGenerationPrompt
				: TITLE_GENERATION.DEFAULT_PROMPT;
		const titlePrompt = titlePromptTemplate
			.replace('{{USER}}', String(userContent || ''))
			.replace('{{ASSISTANT}}', String(assistantContent || ''));
		const titleMessage: ApiChatMessageData = {
			content: titlePrompt,
			role: MessageRole.USER
		};
		const titleResponse = await ChatService.generateTitle(titleMessage, effectiveModel);

		if (!titleResponse) {
			return;
		}

		let cleanTitle = titleResponse.trim();

		cleanTitle = cleanTitle
			.replace(TITLE_GENERATION.PREFIX_PATTERN, '')
			.replace(TITLE_GENERATION.QUOTE_PATTERN, '')
			.trim();

		if (!cleanTitle || cleanTitle.length < TITLE_GENERATION.MIN_LENGTH) {
			const firstLine = userContent.split('\n').find((l) => l.trim().length > 0);

			cleanTitle = firstLine ? firstLine.trim() : TITLE_GENERATION.FALLBACK;
		}

		if (cleanTitle && cleanTitle.length >= TITLE_GENERATION.MIN_LENGTH) {
			await conversationsStore.updateConversationName(convId, cleanTitle);
		}
	}

	private getChatStreamingState(
		convId: string
	): { response: string; messageId: string } | undefined {
		return this.chatStreamingStates.get(convId);
	}

	/**
	 * Pre-encodes the branch ending at the leaf as the next turn reads it: the
	 * stored path cut at its last compaction, with the tools, message shaping
	 * and reasoning handling of the chat, so the server warms the very prefix
	 * that turn extends. Its prompt timings, or null when it returned none.
	 */
	private async preEncodeBranch(
		convId: string,
		leafId: string,
		model: string | null | undefined,
		signal: AbortSignal
	): Promise<ChatMessageTimings | null> {
		const allMessages = await conversationsStore.getConversationMessages(convId);
		const branchPath = [...filterByLeafNodeId(allMessages, leafId, false)];
		const request = await agenticStore.requestFor(
			await ChatService.normalizeMessagesForApi(sliceAtLastCompaction(branchPath)),
			this.toolPolicy()
		);

		return ChatService.preEncode(
			request.messages,
			model,
			Boolean(settingsStore.config.excludeReasoningFromContext),
			signal,
			request.tools
		);
	}

	private async savePartialResponseIfNeeded(convId?: string): Promise<void> {
		const conversationId = convId || conversationsStore.activeConversation?.id;

		if (!conversationId) return;

		const streamingState = this.getChatStreamingState(conversationId);

		if (!streamingState) return;

		const messages =
			conversationId === conversationsStore.activeConversation?.id
				? conversationsStore.activeMessages
				: await conversationsStore.getConversationMessages(conversationId);

		if (!messages.length) return;

		const lastMessage = messages[messages.length - 1];

		if (lastMessage?.role !== MessageRole.ASSISTANT) return;

		const partialContent = streamingState.response;
		const partialReasoning = lastMessage.reasoningContent || '';
		// snapshot the streamed tool calls before clearing so we still know whether
		// anything was captured when deciding to skip the DB write below
		const hadPartialToolCalls = !!lastMessage.toolCalls?.trim();

		// nothing to persist when content, reasoning, and streamed tool calls are all empty
		// (e.g. stop before any token). otherwise drop the partial tool call and write whatever
		// was streamed: incomplete arguments (truncated JSON, missing closing quote) would
		// otherwise be re-sent to the server on the next turn and rejected.
		if (!partialContent.trim() && !partialReasoning.trim() && !hadPartialToolCalls) return;

		try {
			const updateData: {
				content?: string;
				reasoningContent?: string;
				toolCalls?: string;
				timings?: ChatMessageTimings;
			} = {
				toolCalls: ''
			};

			if (partialContent.trim()) updateData.content = partialContent;

			if (partialReasoning.trim()) updateData.reasoningContent = partialReasoning;

			const lastKnownState = this.processing.getState(conversationId);

			if (lastKnownState) {
				updateData.timings = {
					cache_n: lastKnownState.cacheTokens || 0,
					predicted_ms:
						lastKnownState.tokensPerSecond && lastKnownState.tokensDecoded
							? (lastKnownState.tokensDecoded / lastKnownState.tokensPerSecond) * 1000
							: undefined,
					predicted_n: lastKnownState.tokensDecoded || 0,
					prompt_ms: lastKnownState.promptMs,
					prompt_n: lastKnownState.promptTokens || 0
				};
			}

			await DatabaseService.updateMessage(lastMessage.id, updateData);
			lastMessage.content = partialContent;
			// mirror the drop into the in-memory message so the next request sent via
			// sendMessage (queued pending, Send immediately, or manual follow-up) reads
			// the cleared value, not whatever the streaming widget had been showing
			lastMessage.toolCalls = '';

			if (updateData.timings) lastMessage.timings = updateData.timings;
		} catch (error) {
			lastMessage.content = partialContent;
			lastMessage.toolCalls = '';
			console.error('Failed to save partial response:', error);
		}
	}

	// first messages of a new conversation: root, system prompt, working directory. Returns the parent for the first user message
	private async startNewConversation(conv: DatabaseConversation): Promise<string> {
		const rootId = await DatabaseService.createRootMessage(conv.id);
		const systemPrompt = settingsStore.config.systemMessage?.toString().trim();

		let sysOrRootId = rootId;

		// a hat brings its own system text
		if (systemPrompt && !conv.hat) {
			const systemMessage = await DatabaseService.createSystemMessage(
				conv.id,
				systemPrompt,
				rootId
			);

			conversationsStore.addMessageToActive(systemMessage);
			sysOrRootId = systemMessage.id;
		}

		// Reflect a working directory picked on the new-chat screen into
		// chat history before the first user message, so the model sees
		// it on its first turn. createConversation() has already threaded
		// the pending pick onto the conversation.
		if (conv.cwd) {
			const cwdMessage = await this.addMessage(
				MessageRole.USER,
				formatCwdMessage(conv.cwd, await toolsStore.resolveServerHome()),
				MessageType.TEXT,
				sysOrRootId,
				undefined,
				true
			);

			return cwdMessage.id;
		}

		return sysOrRootId;
	}

	/**
	 * Re-send a message queued while `convId` was busy. sendMessage targets the
	 * active conversation, so if the user navigated away the message stays
	 * queued on its own conversation until flushPendingMessage picks it up.
	 */
	private async sendQueuedMessage(
		convId: string,
		msg: { content: string; extras?: DatabaseMessageExtra[] }
	): Promise<void> {
		if (conversationsStore.activeConversation?.id !== convId) {
			this.pendingMessages.set(convId, msg);

			return;
		}

		await this.sendMessage(msg.content, msg.extras);
	}

	/**
	 * Streams a fresh summary of `source` into the compaction node and re-stamps
	 * the reduced context afterwards. `rollback` restores the row when nothing
	 * usable came back, so an empty or aborted summary never replaces what the
	 * node already held.
	 */
	private async summarizeIntoNode(
		convId: string,
		node: DatabaseMessage,
		source: DatabaseMessage[],
		rollback: () => Promise<void>
	): Promise<void> {
		const abortController = this.getOrCreateAbortController(convId);
		const compactionStreamId = COMPACTION.STREAM_ID_PREFIX + convId;
		// Model resolution mirrors the request funnel: the selected model
		// first, then the model the branch was generated with, so router
		// mode targets the child that owns this conversation's KV cache.
		const effectiveModel = serverStore.isRouterMode
			? modelsStore.selectedModelName || getConversationModel(source)
			: undefined;

		let summary = '';
		// a summary the output limit cut is no summary
		let truncated = false;
		let summaryModel = node.model;
		let summaryTimings: ChatMessageCompactionTimings | undefined;

		const writeNode = (updates: Partial<DatabaseMessage>) => {
			if (conversationsStore.activeConversation?.id !== convId) return;

			const idx = conversationsStore.findMessageIndex(node.id);

			if (idx >= 0) conversationsStore.updateMessageAtIndex(idx, updates);
		};

		// The node is the row that reports this run's progress, so the turns it
		// summarizes stay quiet for as long as it runs.
		this.activity.markCompacting(convId, node.id);
		// Clearing the stored summary first hands the row over to the pending
		// state, where it reports the prompt progress instead of the stale text.
		writeNode({ content: '' });

		// The steps the compaction keeps are part of the source the caller
		// handed over; the summarization itself reads only what lies before
		// them, so the summary describes the steps the model will still see.
		const preserve = this.compactionPreserve();
		const cut = compactionCut(source, preserve);

		if (cut === null) {
			await rollback();

			return;
		}

		try {
			// The request is the one the chat sends, its options, tools and
			// message shaping, cut before the preserved steps: the server
			// reuses the prefix it cached and only the instruction is new.
			const request = await agenticStore.requestFor(
				await ChatService.normalizeMessagesForApi(source.slice(0, cut)),
				this.toolPolicy()
			);
			const configValue = settingsStore.config;
			const compactionPrompt =
				typeof configValue.compactionPrompt === 'string' && configValue.compactionPrompt.trim()
					? configValue.compactionPrompt
					: COMPACTION.DEFAULT_PROMPT;

			await ChatService.sendMessage(
				[...request.messages, { content: compactionPrompt, role: MessageRole.USER }],
				{
					...this.getApiOptions(),
					model: effectiveModel,
					onChunk: (chunk: string) => {
						summary += chunk;
						writeNode({ content: summary });
					},
					onComplete: (
						_content: string,
						_reasoning?: string,
						_timings?: ChatMessageTimings,
						_toolCalls?: string,
						finishReason?: string
					) => {
						truncated = finishReason === COMPACTION.FINISH_CUT;
					},
					onModel: (modelName: string) => {
						summaryModel = modelName;
						writeNode({ model: modelName });
					},
					// The summary streams through the same pipe as a chat turn, which
					// enables prompt progress on the wire; feeding it to the shared
					// processing state lets the pending row show the percentage. The
					// closing event carries the timings the row reports once done.
					onTimings: (timings?: ChatMessageTimings, promptProgress?: ChatMessagePromptProgress) => {
						if (timings) summaryTimings = timings;

						this.processing.applyStreamTimings(timings, promptProgress, convId);
					},
					stream: true,
					timings_per_token: true,
					tools: request.tools
				},
				// Registered under the prefixed identity: the visibility kick
				// and byte-offset resume protect the summary through tab
				// switches, while the reload probe for the bare conversation
				// id never reattaches it as a chat turn.
				compactionStreamId,
				abortController.signal
			);

			if (truncated) throw new Error('The output limit cut the compaction summary');

			if (!summary.trim()) {
				await rollback();

				return;
			}

			const stored: Partial<DatabaseMessage> = { content: summary, kept: preserve };

			if (summaryModel) stored.model = summaryModel;

			writeNode(stored);
			await DatabaseService.updateMessage(node.id, stored);

			// Re-encoding the reduced prefix is part of the compaction
			// itself: it rebuilds the server KV cache on the summary and
			// stamps the true context size on the node for the gauge,
			// independently of the per-turn pre-fill setting.
			void this.triggerCompactionPreEncode(convId, node.id, effectiveModel, summaryTimings);
		} catch (error) {
			await rollback();

			if (isAbortError(error)) {
				// The server keeps generating a detached summary otherwise.
				void ChatService.cancelServerStream(compactionStreamId, effectiveModel);
			} else {
				console.error('Failed to compact conversation:', error);
				this.showErrorDialog({
					message: error instanceof Error ? error.message : 'Unknown error',
					type: ErrorDialogType.SERVER
				});
			}
		} finally {
			this.abortControllers.delete(convId);
			// Nothing infers for this conversation anymore: the summarization
			// progress reported above must not outlive the request, or a
			// skipped pre-encode would leave the gauge on a live state that
			// describes the pre-compaction prompt.
			this.processing.setState(convId, null);
			this.setChatLoading(convId, false);
			// Released last: ending it after the loading flag keeps the chat
			// turns from flashing their own indicator for a tick.
			this.activity.endCompacting(convId);
		}
	}

	/** The tool policy of the active conversation, the one its requests run under. */
	private toolPolicy(): AgenticToolPolicy {
		return {
			disabledToolCategories: conversationsStore.preferences.getDisabledToolCategories(),
			disabledTools: conversationsStore.preferences.getDisabledTools()
		};
	}

	/**
	 * Pre-encodes the compacted branch and stamps the returned prompt
	 * timings on the compaction node, so the context gauge reflects the
	 * reduced context as soon as the compaction finishes instead of after
	 * the next turn.
	 */
	private async triggerCompactionPreEncode(
		convId: string,
		nodeId: string,
		model?: string | null,
		summaryTimings?: ChatMessageCompactionTimings
	): Promise<void> {
		this.cancelPreEncode();
		this.preEncodeAbortController = new AbortController();

		const signal = this.preEncodeAbortController.signal;

		try {
			const allIdle = await ChatService.areAllSlotsIdle(model, signal);

			if (!allIdle || signal.aborted) {
				if (!signal.aborted) {
					console.warn('[ChatStore] Compaction pre-encode skipped: slots busy');
				}

				return;
			}

			const timings = await this.preEncodeBranch(convId, nodeId, model, signal);

			if (!timings || signal.aborted) {
				if (!signal.aborted) {
					console.warn('[ChatStore] Compaction pre-encode returned no timings');
				}

				return;
			}

			// The row reports both what the summary cost (summaryTimings) and the
			// size of the context it left behind (timings), so the second is
			// nested under the first instead of overwriting it.
			const stamped = summaryTimings ? { ...timings, compaction: summaryTimings } : timings;

			await DatabaseService.updateMessage(nodeId, { timings: stamped });

			if (conversationsStore.activeConversation?.id === convId) {
				const idx = conversationsStore.findMessageIndex(nodeId);

				if (idx >= 0) conversationsStore.updateMessageAtIndex(idx, { timings: stamped });
			}

			// Nothing is inferring anymore: clearing the live processing
			// state hands the gauge over to the stamped node, which any
			// later restore also picks as the context truth.
			this.processing.setState(convId, null);
		} catch (err) {
			if (!isAbortError(err)) {
				console.warn('[ChatStore] Compaction pre-encode failed:', err);
			}
		}
	}

	/**
	 * Pre-encodes the branch ending at the leaf when every slot is idle, so
	 * the next turn finds its prefix already in the server KV cache.
	 */
	private async triggerPreEncode(
		convId: string,
		leafId: string,
		model?: string | null
	): Promise<void> {
		this.cancelPreEncode();
		this.preEncodeAbortController = new AbortController();

		const signal = this.preEncodeAbortController.signal;

		try {
			const allIdle = await ChatService.areAllSlotsIdle(model, signal);

			if (!allIdle || signal.aborted) return;

			await this.preEncodeBranch(convId, leafId, model, signal);
		} catch (err) {
			if (!isAbortError(err)) {
				console.warn('[ChatStore] Pre-encode failed:', err);
			}
		}
	}

	/**
	 * The hat this conversation wears, with its prompt cache restored on the
	 * server when this request could not hit it otherwise. Null for no hat.
	 */
	private async wearHat(convId: string, signal: AbortSignal): Promise<HatWear | null> {
		const conv =
			conversationsStore.activeConversation?.id === convId
				? conversationsStore.activeConversation
				: conversationsStore.conversations.find((c) => c.id === convId);

		if (!conv?.hat) return null;

		if (!hatsStore.available) {
			hatsStore.warnMissing(conv.hat);

			return null;
		}

		return hatsStore.prepare(convId, conv.hat, signal);
	}
}

export const chatStore = new ChatStore();
