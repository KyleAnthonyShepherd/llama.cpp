<script lang="ts">
	import { GitFork, ShieldAlert } from '@lucide/svelte';
	import { MarkdownContent } from '$lib/components/app';
	import { MessageRole } from '$lib/enums';
	import { chatStore, conversationsStore, hydraStore } from '$lib/stores';
	import { untrack } from 'svelte';

	interface Props {
		class?: string;
		/** the tail message of the head on the active path */
		message: DatabaseMessage;
	}

	let { class: className = '', message }: Props = $props();

	let all = $state<DatabaseMessage[]>([]);

	// reload when the path moves or a reply starts or ends (the live map gains or loses entries)
	$effect(() => {
		void message.id;
		void hydraStore.live.size;
		void conversationsStore.activeMessages.length;

		untrack(() => {
			conversationsStore.getConversationMessages(message.convId).then((m) => (all = m));
		});
	});

	// every head: its tail, then its chain of replies and loop guard notes
	const heads = $derived.by(() => {
		const group = message.hydra?.group;

		if (!group) return [];

		return all
			.filter((m) => m.parent === message.parent && m.hydra?.group === group)
			.sort((a, b) => (a.hydra?.index ?? 0) - (b.hydra?.index ?? 0))
			.map((tail) => {
				const chain: DatabaseMessage[] = [];

				for (let cur = tail; ; ) {
					const next = all.find((m) => m.parent === cur.id && m.hydra?.group === group);

					if (!next) break;

					chain.push(next);
					cur = next;
				}

				return { chain, tail };
			});
	});

	const isLoading = $derived(chatStore.isChatLoading(message.convId));
	const rootProgress = $derived(hydraStore.rootProgress.get(message.convId));

	function speed(timings?: ChatMessageTimings): string {
		if (!timings?.predicted_n || !timings.predicted_ms) return '';

		return `${timings.predicted_n} tokens, ${((timings.predicted_n * 1000) / timings.predicted_ms).toFixed(1)} t/s`;
	}

	function percent(p?: ChatMessagePromptProgress): string {
		if (!p || !p.total) return '';

		return `${Math.round(((p.cache + p.processed) * 100) / p.total)}%`;
	}
</script>

<div class={['hydra-group', className]}>
	<div class="mb-2 flex items-center gap-2 text-xs text-muted-foreground">
		<GitFork class="h-3.5 w-3.5" />

		<span>Hydra, {heads.length} heads on one root</span>

		{#if isLoading && rootProgress && rootProgress.processed + rootProgress.cache < rootProgress.total}
			<span>- reading the root {percent(rootProgress)}</span>
		{/if}
	</div>

	<!-- heads wrap into rows, so many heads stay reachable with the page scroll -->
	<div class="grid grid-cols-[repeat(auto-fill,minmax(17rem,1fr))] gap-3 pb-2">
		{#each heads as { chain, tail } (tail.id)}
			{@const last = chain.findLast((m) => m.role === MessageRole.ASSISTANT)}
			{@const lastLive = last ? hydraStore.live.get(last.id) : undefined}
			{@const notes = chain.filter((m) => m.role === MessageRole.SYSTEM).length}
			{@const current = tail.id === message.id}

			<div
				class={[
					'flex min-w-0 flex-col rounded-2xl border p-3',
					current ? 'border-primary/50 bg-muted/20' : 'border-border/50'
				]}
			>
				<div class="mb-2 flex items-center gap-2 text-xs">
					<span class="font-medium">Head {(tail.hydra?.index ?? 0) + 1}</span>

					{#if lastLive && !lastLive.done}
						<span class="text-muted-foreground">
							{#if lastLive.reflecting}
								reflecting...
							{:else if lastLive.content || lastLive.reasoning}
								writing...
							{:else}
								{`reading ${percent(lastLive.progress) || '...'}`}
							{/if}
						</span>
					{:else}
						<span class="text-muted-foreground">{speed(lastLive?.timings ?? last?.timings)}</span>
					{/if}

					{#if notes > 0}
						<span
							class="flex items-center gap-0.5 text-amber-600 dark:text-amber-400"
							title="The loop guard stepped in on this head"
						>
							<ShieldAlert class="h-3 w-3" />
							{notes}
						</span>
					{/if}

					{#if current}
						<span class="ml-auto text-muted-foreground">this branch</span>
					{:else}
						<button
							class="ml-auto rounded-md px-2 py-0.5 text-muted-foreground hover:bg-accent hover:text-foreground disabled:opacity-40"
							disabled={isLoading}
							onclick={() => conversationsStore.navigateToSibling(tail.id)}
							title="Show this head's branch and continue the chat from its reply"
							type="button"
						>
							Continue here
						</button>
					{/if}
				</div>

				<div class="mb-2 rounded-xl bg-muted/40 px-3 py-2 text-sm whitespace-pre-wrap">
					{tail.content}
				</div>

				<div class="max-h-[32rem] min-w-0 overflow-y-auto text-sm">
					{#each chain as msg (msg.id)}
						{#if msg.role === MessageRole.SYSTEM}
							<div
								class="my-2 rounded-lg border-l-2 border-amber-500/70 bg-amber-500/5 px-2 py-1.5 text-xs text-muted-foreground"
							>
								<div class="mb-0.5 font-medium text-amber-700 dark:text-amber-400">
									Loop guard note
								</div>

								<div class="whitespace-pre-wrap">{msg.content}</div>
							</div>
						{:else}
							{@const live = hydraStore.live.get(msg.id)}
							{@const content = live?.content ?? msg.content}
							{@const reasoning = live?.reasoning ?? msg.reasoningContent ?? ''}

							{#if reasoning}
								<details class="mb-2 text-xs text-muted-foreground">
									<summary class="cursor-pointer">Reasoning</summary>

									<div class="mt-1 whitespace-pre-wrap">{reasoning}</div>
								</details>
							{/if}

							{#if content}
								<MarkdownContent {content} />
							{:else if !live}
								<span class="text-xs text-muted-foreground">(no reply)</span>
							{/if}

							{#if live?.reflecting || live?.reflection}
								<div
									class="my-2 rounded-lg border border-dashed border-amber-500/50 px-2 py-1.5 text-xs text-muted-foreground"
								>
									<div class="mb-0.5 font-medium">Loop guard reflection</div>

									<div class="whitespace-pre-wrap italic">{live.reflection || '...'}</div>
								</div>
							{/if}
						{/if}
					{/each}
				</div>
			</div>
		{/each}
	</div>
</div>
