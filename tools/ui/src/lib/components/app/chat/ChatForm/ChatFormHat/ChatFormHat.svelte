<script lang="ts">
	import { Check, GraduationCap, Pencil, Plus } from '@lucide/svelte';
	import { DialogHatEditor } from '$lib/components/app';
	import * as Popover from '$lib/components/ui/popover';
	import { HAT_STATE_LABELS } from '$lib/constants';
	import { MessageRole } from '$lib/enums';
	import { conversationsStore, hatsStore } from '$lib/stores';
	import type { HatSummary } from '$lib/types';
	import { untrack } from 'svelte';

	interface Props {
		class?: string;
		disabled?: boolean;
	}

	let { class: className = '', disabled = false }: Props = $props();

	let editorName = $state<string | null>(null);
	let editorOpen = $state(false);
	let isOpen = $state(false);

	const hat = $derived(conversationsStore.preferences.getHat());
	// the cache is the start of the prompt: a chat that already has replies would be read again from the top
	const locked = $derived(
		conversationsStore.activeMessages.some((m) => m.role === MessageRole.ASSISTANT)
	);

	// list the hats once the server's tools are known
	$effect(() => {
		if (hatsStore.available) untrack(() => void hatsStore.refresh());
	});

	function stateNote(summary: HatSummary): string {
		if (summary.build?.running) return 'building';

		const size = summary.tokens ? `${summary.tokens.toLocaleString()} tokens` : '';
		const state =
			summary.state === 'ready' ? '' : (HAT_STATE_LABELS[summary.state] ?? summary.state);

		return [size, state].filter(Boolean).join(', ');
	}

	async function pick(name: string | null) {
		await conversationsStore.preferences.setHat(name);
		isOpen = false;
	}

	function edit(name: string | null) {
		isOpen = false;
		editorName = name;
		editorOpen = true;
	}

	async function handleEditorChange(name: string | null) {
		await hatsStore.refresh();

		// a renamed or deleted hat must not stay picked under its old name
		if (!locked && editorName !== null && hat === editorName) {
			await conversationsStore.preferences.setHat(name);
		}

		editorName = name;
	}
</script>

{#snippet row(name: string | null, title: string, note: string, editable: boolean)}
	<div class="flex items-center gap-1 rounded-sm hover:bg-accent">
		<button
			class="flex min-w-0 flex-1 items-center gap-2 px-2 py-1.5 text-left text-sm disabled:opacity-50"
			disabled={locked && name !== hat}
			onclick={() => pick(name)}
			type="button"
		>
			<Check class={['h-4 w-4 shrink-0', name === hat ? 'opacity-100' : 'opacity-0']} />

			<span class="min-w-0 flex-1">
				<span class="block truncate">{title}</span>

				{#if note}
					<span class="block truncate text-xs text-muted-foreground">{note}</span>
				{/if}
			</span>
		</button>

		{#if editable}
			<button
				aria-label="Edit hat {title}"
				class="p-1.5 text-muted-foreground hover:text-foreground"
				onclick={() => edit(name)}
				type="button"
			>
				<Pencil class="h-3.5 w-3.5" />
			</button>
		{/if}
	</div>
{/snippet}

{#if hatsStore.available}
	<Popover.Root
		onOpenChange={(open) => {
			isOpen = open;

			if (open) void hatsStore.refresh();
		}}
		open={isOpen}
	>
		<Popover.Trigger
			class={[
				'flex min-w-0 w-auto items-center gap-1 mt-1.5 py-1 px-2 backdrop-blur-2xl rounded-md text-xs cursor-pointer',
				hat ? 'text-foreground' : 'text-muted-foreground',
				className
			]}
			{disabled}
		>
			<GraduationCap class="w-3.5 h-3.5" />

			<span class="max-w-48 truncate">{hat ?? 'No hat'}</span>
		</Popover.Trigger>

		<Popover.Content
			align="start"
			class="w-80 rounded-xl border-border/50 p-1 shadow-xl"
			side="top"
		>
			{@render row(null, 'No hat', 'your own system prompt and tool choices', false)}

			{#each hatsStore.hats as summary (summary.name)}
				{@render row(
					summary.name,
					summary.name,
					[summary.description, stateNote(summary)].filter(Boolean).join(' - '),
					true
				)}
			{/each}

			{#if hatsStore.error}
				<div class="px-2 py-1.5 text-xs text-destructive">{hatsStore.error}</div>
			{/if}

			{#if locked}
				<div class="px-2 py-1.5 text-xs text-muted-foreground">
					A chat keeps the hat it started with.
				</div>
			{/if}

			<div aria-hidden="true" class="my-1 h-px bg-border/40"></div>

			<button
				class="flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-sm hover:bg-accent"
				onclick={() => edit(null)}
				type="button"
			>
				<Plus class="h-4 w-4" />
				New hat
			</button>
		</Popover.Content>
	</Popover.Root>

	<DialogHatEditor
		bind:open={editorOpen}
		name={editorName}
		onChange={handleEditorChange}
		onClose={() => hatsStore.refresh()}
	/>
{/if}
