<script lang="ts">
	import { ChevronDown, Copy, GitFork, Minus, Plus, X } from '@lucide/svelte';
	import { hydraStore } from '$lib/stores';

	interface Props {
		class?: string;
		disabled?: boolean;
	}

	let { class: className = '', disabled = false }: Props = $props();

	const count = $derived(hydraStore.tails.length);

	const THINKING = [
		['default', 'Default'],
		['on', 'On'],
		['off', 'Off']
	] as const;

	// on: a click on the collapsed chip opens the panel again; off: the chip turns hydra on
	function handleChipClick() {
		if (hydraStore.enabled && hydraStore.collapsed) {
			hydraStore.collapsed = false;
		} else {
			hydraStore.enabled = !hydraStore.enabled;
			hydraStore.collapsed = false;
		}
	}
</script>

{#if hydraStore.available}
	<button
		aria-pressed={hydraStore.enabled}
		class={[
			'flex w-auto min-w-0 cursor-pointer items-center gap-1 mt-1.5 py-1 px-2 backdrop-blur-2xl rounded-md text-xs',
			hydraStore.enabled ? 'bg-accent text-foreground' : 'text-muted-foreground',
			className
		]}
		{disabled}
		onclick={handleChipClick}
		title="Hydra: one root (the chat so far and the message box), several heads that each add their own message and reply side by side"
		type="button"
	>
		<GitFork class="h-3.5 w-3.5" />

		<span>
			{hydraStore.enabled
				? `Hydra: ${count} heads${hydraStore.thinking === 'off' ? ', no thinking' : ''}`
				: 'Hydra'}
		</span>

		{#if hydraStore.enabled && hydraStore.collapsed}
			<ChevronDown class="h-3.5 w-3.5" />
		{/if}
	</button>

	{#if hydraStore.enabled && !hydraStore.collapsed}
		<div
			class="mt-2 basis-full rounded-2xl border border-border/50 bg-background/60 p-3 backdrop-blur-md"
		>
			<div class="mb-2 flex items-center gap-2 text-xs text-muted-foreground">
				<span>Heads</span>

				<button
					aria-label="One head less"
					class="rounded-md p-1 hover:bg-accent disabled:opacity-40"
					disabled={disabled || count <= 1}
					onclick={() => hydraStore.setCount(count - 1)}
					type="button"
				>
					<Minus class="h-3.5 w-3.5" />
				</button>

				<span class="w-5 text-center text-foreground tabular-nums">{count}</span>

				<button
					aria-label="One head more"
					class="rounded-md p-1 hover:bg-accent disabled:opacity-40"
					disabled={disabled || count >= hydraStore.maxHeads}
					onclick={() => hydraStore.setCount(count + 1)}
					type="button"
				>
					<Plus class="h-3.5 w-3.5" />
				</button>

				<span class="text-muted-foreground/70">max {hydraStore.maxHeads}</span>

				<span class="ml-3">Thinking</span>

				<div class="flex overflow-hidden rounded-md border border-border/50">
					{#each THINKING as [value, label] (value)}
						<button
							aria-pressed={hydraStore.thinking === value}
							class={[
								'px-2 py-0.5 hover:bg-accent disabled:opacity-40',
								hydraStore.thinking === value ? 'bg-accent text-foreground' : ''
							]}
							{disabled}
							onclick={() => (hydraStore.thinking = value)}
							title={value === 'default'
								? 'The thinking setting of the chat'
								: `Thinking ${value} for every head`}
							type="button"
						>
							{label}
						</button>
					{/each}
				</div>

				<button
					class="ml-auto flex items-center gap-1 rounded-md px-2 py-1 hover:bg-accent disabled:opacity-40"
					disabled={disabled || !hydraStore.tails[0]?.trim()}
					onclick={() => hydraStore.fillFromFirst()}
					title="Copy the first head's message into the empty boxes"
					type="button"
				>
					<Copy class="h-3.5 w-3.5" />
					Fill empty from head 1
				</button>

				<button
					aria-label="Hide the head boxes"
					class="rounded-md p-1 hover:bg-accent"
					onclick={() => (hydraStore.collapsed = true)}
					title="Hide the head boxes (hydra stays on, the chip opens them again)"
					type="button"
				>
					<X class="h-3.5 w-3.5" />
				</button>
			</div>

			<!-- capped and scrolled: with many heads the form would grow past the top of the window -->
			<div class="grid max-h-[35vh] gap-2 overflow-y-auto pr-1 md:grid-cols-2">
				{#each hydraStore.tails as _, i (i)}
					<div class="flex items-start gap-2">
						<span class="mt-1.5 w-14 shrink-0 text-xs text-muted-foreground">Head {i + 1}</span>

						<textarea
							bind:value={hydraStore.tails[i]}
							class="field-sizing-content max-h-40 min-h-8 w-full resize-none rounded-md bg-muted/30 px-2 py-1 text-sm outline-none placeholder:text-muted-foreground/60 focus-visible:bg-muted/50"
							{disabled}
							placeholder="This head's own message..."
							rows="1"
						></textarea>
					</div>
				{/each}
			</div>

			<p class="mt-2 text-xs text-muted-foreground">
				The message box above is the last message of the shared root (it can stay empty). Every head
				needs its own message.
			</p>
		</div>
	{/if}
{/if}
