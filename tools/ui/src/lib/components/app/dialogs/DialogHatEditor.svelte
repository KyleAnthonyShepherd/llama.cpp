<script lang="ts">
	import { ArrowDown, ArrowUp, FilePlus, Hammer, Square, Trash2, X } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import * as Dialog from '$lib/components/ui/dialog';
	import { Input } from '$lib/components/ui/input';
	import { Textarea } from '$lib/components/ui/textarea';
	import {
		HAT_BUILD_POLL_MS,
		HAT_DOCUMENT_MAX_BYTES,
		HAT_NAME_PATTERN,
		HAT_READER_LABELS,
		HAT_STATE_LABELS
	} from '$lib/constants';
	import { ToolSource } from '$lib/enums';
	import { hatsStore, toolsStore } from '$lib/stores';
	import type { HatDetail, HatDocument, HatSummary, OpenAIToolDefinition } from '$lib/types';
	import { formatFileSize } from '$lib/utils';
	import { untrack } from 'svelte';
	import { toast } from 'svelte-sonner';

	interface Props {
		/** Hat to edit, null to make a new one */
		name: string | null;
		open: boolean;
		onClose: () => void;
		/** Fired after a hat was created, renamed or deleted, with its name (null when deleted) */
		onChange?: (name: string | null) => void;
	}

	let { name, onChange, onClose, open = $bindable() }: Props = $props();

	let busy = $state(false);
	let confirmDelete = $state(false);
	let description = $state('');
	let detail = $state<HatDetail | null>(null);
	let documents = $state<HatDocument[]>([]);
	let fileInput = $state<HTMLInputElement | null>(null);
	let hatName = $state('');
	let instructions = $state('');
	let status = $state<HatSummary | null>(null);
	let tools = $state<string[]>([]);

	const building = $derived(!!status?.build?.running);
	const nameValid = $derived(HAT_NAME_PATTERN.test(hatName));
	const contextSize = $derived(hatsStore.server?.n_ctx ?? 0);
	const instructionsAndText = $derived(
		instructions.length + documents.reduce((sum, d) => sum + (d.chars ?? 0), 0)
	);
	// before a build only the text is known: about four characters to a token
	const tokens = $derived(
		status?.state === 'ready' && status.tokens ? status.tokens : Math.round(instructionsAndText / 4)
	);
	const dirty = $derived(
		detail !== null &&
			(hatName !== detail.name ||
				description !== detail.description ||
				instructions !== detail.instructions ||
				JSON.stringify(tools) !== JSON.stringify(detail.tool_names) ||
				JSON.stringify(documents.map((d) => [d.file, d.reader, d.title])) !==
					JSON.stringify(detail.document_list.map((d) => [d.file, d.reader, d.title])))
	);
	const unread = $derived(documents.filter((d) => !d.read).length);
	const progress = $derived(
		status?.build && status.build.total > 0
			? Math.round((100 * status.build.done) / status.build.total)
			: null
	);

	// start from the hat on the server each time the dialog opens or is pointed at another hat
	$effect(() => {
		const hat = name;

		if (!open) return;

		untrack(() => {
			confirmDelete = false;
			void hatsStore.refresh();

			if (hat) {
				void load(hat);
			} else {
				detail = null;
				status = null;
				hatName = '';
				description = '';
				instructions = '';
				tools = [];
				documents = [];
			}
		});
	});

	// follow a running build
	$effect(() => {
		if (!open || !building || !detail) return;

		const current = detail.name;
		const timer = setInterval(async () => {
			try {
				status = await hatsStore.status(current);

				if (!status.build?.running) await load(current, true);
			} catch {
				// a missed poll is not worth a message, the next one tells
			}
		}, HAT_BUILD_POLL_MS);

		return () => clearInterval(timer);
	});

	function adopt(next: HatDetail, keepEdits = false) {
		detail = next;
		status = next;

		if (keepEdits) {
			// keep what is being typed; take what only the server knows
			const known = new Map(next.document_list.map((d) => [d.file, d]));

			documents = [
				...documents
					.filter((d) => known.has(d.file))
					.map((d) => ({ ...known.get(d.file)!, reader: d.reader, title: d.title })),
				...next.document_list.filter((d) => !documents.some((mine) => mine.file === d.file))
			];

			return;
		}

		hatName = next.name;
		description = next.description;
		instructions = next.instructions;
		tools = [...next.tool_names];
		documents = next.document_list.map((d) => ({ ...d }));
	}

	async function run(action: () => Promise<void>) {
		if (busy) return;

		busy = true;

		try {
			await action();
		} catch (err) {
			toast.error(err instanceof Error ? err.message : String(err), { duration: 10000 });
		} finally {
			busy = false;
		}
	}

	async function load(hat: string, keepEdits = false) {
		try {
			adopt(await hatsStore.get(hat), keepEdits);
		} catch (err) {
			toast.error(err instanceof Error ? err.message : String(err));
		}
	}

	function toolDefinitions(): Record<string, OpenAIToolDefinition> {
		// the server knows its own tools; the ones that live in the browser go along with the hat
		const defs: Record<string, OpenAIToolDefinition> = {};

		for (const entry of toolsStore.allTools) {
			const toolName = entry.definition.function.name;

			if (entry.source !== ToolSource.SERVER && tools.includes(toolName)) {
				defs[toolName] = entry.definition;
			}
		}

		return defs;
	}

	async function save() {
		if (!detail) return;

		const saved = await hatsStore.save(detail.name, {
			description,
			documents: documents.map((d) => ({ file: d.file, reader: d.reader, title: d.title })),
			instructions,
			rename_to: hatName !== detail.name ? hatName : undefined,
			tool_defs: toolDefinitions(),
			tools
		});
		const renamed = saved.name !== detail.name;

		adopt(saved);

		if (renamed) onChange?.(saved.name);
	}

	function create() {
		void run(async () => {
			adopt(await hatsStore.save(hatName, { create: true, description }));
			onChange?.(hatName);
		});
	}

	function build() {
		void run(async () => {
			if (dirty) await save();

			if (!detail) return;

			await hatsStore.build(detail.name);
			status = await hatsStore.status(detail.name);
		});
	}

	function stop() {
		void run(async () => {
			if (detail) await hatsStore.cancel(detail.name);
		});
	}

	function addFiles(files: FileList | null) {
		void run(async () => {
			if (!detail || !files) return;

			if (dirty) await save();

			for (const file of Array.from(files)) {
				if (file.size > HAT_DOCUMENT_MAX_BYTES) {
					toast.error(`"${file.name}" is larger than ${formatFileSize(HAT_DOCUMENT_MAX_BYTES)}`);

					continue;
				}

				adopt(await hatsStore.putDocument(detail.name, file));
			}

			if (fileInput) fileInput.value = '';
		});
	}

	function removeDocument(file: string) {
		void run(async () => {
			if (!detail) return;

			if (dirty) await save();

			adopt(await hatsStore.removeDocument(detail.name, file));
		});
	}

	function move(index: number, by: number) {
		const to = index + by;

		if (to < 0 || to >= documents.length) return;

		const next = [...documents];

		[next[index], next[to]] = [next[to], next[index]];
		documents = next;
	}

	function toggleTool(toolName: string) {
		tools = tools.includes(toolName) ? tools.filter((t) => t !== toolName) : [...tools, toolName];
	}

	function remove() {
		void run(async () => {
			if (!detail) return;

			await hatsStore.remove(detail.name);
			onChange?.(null);
			close();
		});
	}

	function close() {
		open = false;
		onClose();
	}

	function readNote(doc: HatDocument): string {
		if (!doc.read) return 'not read yet';

		if (doc.pages) {
			return `${doc.pages} pages: ${doc.by_text ?? 0} by text, ${doc.by_vision ?? 0} by vision`;
		}

		return `${doc.chars ?? 0} characters`;
	}
</script>

{#snippet label(text: string, hint?: string)}
	<div class="flex items-baseline justify-between gap-2">
		<span class="text-sm font-medium">{text}</span>

		{#if hint}
			<span class="text-xs text-muted-foreground">{hint}</span>
		{/if}
	</div>
{/snippet}

<Dialog.Root
	onOpenChange={(next) => {
		if (!next) close();
	}}
	{open}
>
	<Dialog.Content class="flex max-h-[90dvh] flex-col gap-0 p-0 sm:max-w-3xl">
		<Dialog.Header class="border-b border-border/50 px-6 py-4">
			<Dialog.Title>{detail ? `Hat: ${detail.name}` : 'New hat'}</Dialog.Title>

			<Dialog.Description>
				A hat is a system prompt, tools and documents the model has already read. A chat that wears
				it starts with all three in the cache.
			</Dialog.Description>
		</Dialog.Header>

		<div class="flex-1 space-y-5 overflow-y-auto px-6 py-4">
			<div class="grid gap-3 sm:grid-cols-[14rem_1fr]">
				<div class="space-y-1">
					{@render label('Name')}

					<Input
						bind:value={hatName}
						aria-invalid={hatName.length > 0 && !nameValid}
						autocomplete="off"
						disabled={building}
						placeholder="astm-cement"
						spellcheck={false}
					/>
				</div>

				<div class="space-y-1">
					{@render label('What it is for')}

					<Input bind:value={description} placeholder="ASTM cement and concrete standards" />
				</div>
			</div>

			{#if hatName.length > 0 && !nameValid}
				<p class="text-xs text-destructive">
					Lower-case letters, digits, - and _, starting with a letter or digit, 40 at most.
				</p>
			{/if}

			{#if detail}
				<div class="space-y-1">
					{@render label('Instructions', 'the system prompt, read after the documents')}

					<Textarea
						bind:value={instructions}
						class="min-h-32 font-mono text-xs"
						placeholder="How to answer, what to quote, when to say it is not in the documents."
					/>
				</div>

				<div class="space-y-1">
					{@render label('Documents', 'the model reads them in this order')}

					{#if documents.length === 0}
						<p
							class="rounded-md border border-dashed border-border/60 px-3 py-4 text-sm text-muted-foreground"
						>
							No documents. A hat without any is a system prompt and a tool set.
						</p>
					{/if}

					<ul class="space-y-1">
						{#each documents as doc, index (doc.file)}
							<li class="flex items-center gap-2 rounded-md border border-border/50 px-2 py-1.5">
								<div class="flex flex-col">
									<button
										aria-label="Move up"
										class="text-muted-foreground hover:text-foreground disabled:opacity-30"
										disabled={index === 0}
										onclick={() => move(index, -1)}
										type="button"
									>
										<ArrowUp class="h-3.5 w-3.5" />
									</button>

									<button
										aria-label="Move down"
										class="text-muted-foreground hover:text-foreground disabled:opacity-30"
										disabled={index === documents.length - 1}
										onclick={() => move(index, 1)}
										type="button"
									>
										<ArrowDown class="h-3.5 w-3.5" />
									</button>
								</div>

								<div class="min-w-0 flex-1">
									<div class="truncate text-sm" title={doc.file}>{doc.file}</div>

									<div class="text-xs text-muted-foreground">
										{formatFileSize(doc.size)} - {readNote(doc)}
									</div>

									{#each doc.notes ?? [] as note (note)}
										<div class="text-xs text-amber-600 dark:text-amber-400">{note}</div>
									{/each}
								</div>

								<select
									bind:value={doc.reader}
									aria-label="How to read {doc.file}"
									class="h-8 rounded-md border border-border/60 bg-background px-2 text-xs"
								>
									{#each hatsStore.readers as reader (reader)}
										<option value={reader}>{HAT_READER_LABELS[reader] ?? reader}</option>
									{/each}
								</select>

								<button
									aria-label="Remove {doc.file}"
									class="text-muted-foreground hover:text-destructive"
									disabled={busy || building}
									onclick={() => removeDocument(doc.file)}
									type="button"
								>
									<X class="h-4 w-4" />
								</button>
							</li>
						{/each}
					</ul>

					<input
						bind:this={fileInput}
						accept={hatsStore.accepts.join(',')}
						class="hidden"
						multiple
						onchange={() => addFiles(fileInput?.files ?? null)}
						type="file"
					/>

					<Button
						disabled={busy || building}
						onclick={() => fileInput?.click()}
						size="sm"
						type="button"
						variant="outline"
					>
						<FilePlus class="h-4 w-4" />
						Add documents
					</Button>
				</div>

				<div class="space-y-1">
					{@render label(
						`Tools (${tools.length})`,
						'tools come first in the prompt: changing them rebuilds the whole hat'
					)}

					<div class="grid gap-x-4 gap-y-3 sm:grid-cols-2">
						{#each toolsStore.toolGroups as group (group.key)}
							<div class="space-y-1">
								<div class="text-xs font-medium text-muted-foreground">{group.label}</div>

								{#each group.tools as entry (entry.key)}
									{@const toolName = entry.definition.function.name}

									<label
										class="flex cursor-pointer items-center gap-2 text-sm"
										title={entry.definition.function.description}
									>
										<Checkbox
											checked={tools.includes(toolName)}
											onCheckedChange={() => toggleTool(toolName)}
										/>

										<span class="truncate font-mono text-xs">{toolName}</span>
									</label>
								{/each}
							</div>
						{/each}
					</div>
				</div>
			{/if}
		</div>

		{#if detail && status}
			<div class="space-y-1 border-t border-border/50 px-6 py-3 text-sm">
				<div class="flex flex-wrap items-center gap-x-3 gap-y-1">
					<span class="font-medium">
						{building ? 'building' : (HAT_STATE_LABELS[status.state] ?? status.state)}
					</span>

					<span class="text-muted-foreground">
						{status.state === 'ready' && !dirty ? '' : 'about '}{tokens.toLocaleString()} tokens{contextSize
							? ` of ${contextSize.toLocaleString()} context`
							: ''}
					</span>

					{#if unread > 0}
						<span class="text-muted-foreground">
							plus {unread} document{unread === 1 ? '' : 's'} not read yet
						</span>
					{/if}

					{#if dirty}
						<span class="text-amber-600 dark:text-amber-400">unsaved changes</span>
					{/if}
				</div>

				{#if building && status.build}
					<div class="text-xs text-muted-foreground">
						{status.build.step}{progress !== null ? ` (${progress}%)` : ''}
					</div>

					<div class="h-1 overflow-hidden rounded bg-muted">
						<div style:width="{progress ?? 5}%" class="h-full bg-primary transition-[width]"></div>
					</div>
				{:else if status.build?.error}
					<div class="text-xs text-destructive">Build failed: {status.build.error}</div>
				{:else if status.why.length > 0}
					<div class="text-xs text-muted-foreground">{status.why.join('; ')}</div>
				{/if}

				{#if contextSize > 0 && tokens > contextSize * 0.8}
					<div class="text-xs text-amber-600 dark:text-amber-400">
						This leaves little of the context for the chat itself.
					</div>
				{/if}
			</div>
		{/if}

		<Dialog.Footer class="border-t border-border/50 px-6 py-3 sm:justify-between">
			<div class="flex items-center gap-2">
				{#if detail}
					{#if confirmDelete}
						<Button disabled={busy || building} onclick={remove} size="sm" variant="destructive">
							Delete this hat
						</Button>

						<Button onclick={() => (confirmDelete = false)} size="sm" variant="ghost">Keep</Button>
					{:else}
						<Button
							aria-label="Delete hat"
							disabled={busy || building}
							onclick={() => (confirmDelete = true)}
							size="sm"
							variant="ghost"
						>
							<Trash2 class="h-4 w-4" />
						</Button>
					{/if}
				{/if}
			</div>

			<div class="flex items-center gap-2">
				<Button onclick={close} variant="ghost">Close</Button>

				{#if !detail}
					<Button disabled={busy || !nameValid} onclick={create}>Create</Button>
				{:else}
					<Button
						disabled={busy || building || !dirty || !nameValid}
						onclick={() => run(save)}
						variant="outline"
					>
						Save
					</Button>

					{#if building}
						<Button disabled={busy} onclick={stop} variant="outline">
							<Square class="h-4 w-4" />
							Stop
						</Button>
					{:else}
						<Button disabled={busy || !nameValid} onclick={build}>
							<Hammer class="h-4 w-4" />
							{status?.state === 'ready' && !dirty ? 'Rebuild' : 'Build'}
						</Button>
					{/if}
				{/if}
			</div>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
