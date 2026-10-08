import {
	FILE_INTAKE_EXTENSIONS,
	FILE_INTAKE_FONT_SEPARATOR,
	FILE_INTAKE_MAX_BYTES,
	FILE_INTAKE_NEEDS_FONTS_PREFIX,
	FILE_INTAKE_NOTE_PREFIX,
	FILE_INTAKE_SAVED_PREFIX,
	FONT_INTAKE_EXTENSIONS,
	RESP_TYPE_BASE64
} from '$lib/constants';
import { BuiltInTool, ToolResponseField } from '$lib/enums';
import { ToolsService } from '$lib/services/tools.service';
import { conversationsStore } from '$lib/stores/conversations/index.svelte';
import { toolsStore } from '$lib/stores/tools.svelte';
import { toast } from 'svelte-sonner';

export function isFileIntakeCandidate(filename: string): boolean {
	const lower = filename.toLowerCase();

	return FILE_INTAKE_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

function readFileAsBase64(file: File): Promise<string> {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();

		reader.onload = () => resolve((reader.result as string).split(',')[1] ?? '');
		reader.onerror = () => reject(reader.error);
		reader.readAsDataURL(file);
	});
}

// a new chat has no conversation yet, its working directory is still pending
function currentCwd(): string | undefined {
	return (
		conversationsStore.activeConversation?.cwd ??
		conversationsStore.preferences.pendingCwd ??
		undefined
	);
}

function parseNeededFonts(response: string): string[] {
	const line = response.split('\n').find((l) => l.startsWith(FILE_INTAKE_NEEDS_FONTS_PREFIX));

	if (!line) return [];

	return line
		.slice(FILE_INTAKE_NEEDS_FONTS_PREFIX.length)
		.split(FILE_INTAKE_FONT_SEPARATOR)
		.map((f) => f.trim())
		.filter(Boolean);
}

/** Let the user pick font files and send them to the server, then report what is still missing. */
function pickAndUploadFonts(path: string, filename: string): void {
	const tool = toolsStore.fontIntakeTool;

	if (!tool) return;

	const input = document.createElement('input');

	input.type = 'file';
	input.multiple = true;
	input.accept = FONT_INTAKE_EXTENSIONS.join(',');
	input.onchange = async () => {
		let added = 0;
		let stillNeeded: string[] = [];

		for (const font of Array.from(input.files ?? [])) {
			try {
				const result = await ToolsService.executeTool(
					tool,
					{ base64: await readFileAsBase64(font), name: font.name, path },
					undefined,
					currentCwd()
				);

				if (result.isError) throw new Error(result.content);

				added += 1;
				stillNeeded = parseNeededFonts(result.content);
			} catch (err) {
				const reason = err instanceof Error ? err.message : String(err);

				toast.error(`Could not add "${font.name}": ${reason}`, { duration: 8000 });
			}
		}

		if (added === 0) return;

		if (stillNeeded.length > 0) {
			warnAboutFonts(path, filename, stillNeeded);
		} else {
			toast.success(`Fonts added. The server can now draw "${filename}" as intended.`);
		}
	};
	input.click();
}

function warnAboutFonts(path: string, filename: string, fonts: string[]): void {
	const message = `The server does not have ${fonts.length === 1 ? 'a font' : 'fonts'} "${filename}" uses: ${fonts.join(', ')}. Pictures of it will not match the original layout.`;

	toast.warning(message, {
		action: toolsStore.fontIntakeTool
			? { label: 'Add fonts', onClick: () => pickAndUploadFonts(path, filename) }
			: undefined,
		duration: 30000
	});
}

/**
 * Save an attached file on the server through the file intake tool.
 * The model gets the returned note in place of the file content.
 *
 * @throws if no intake tool is served, the file is too large, or the save fails
 */
export async function saveFileThroughIntake(file: File): Promise<string> {
	const tool = toolsStore.fileIntakeTool;

	if (!tool) {
		throw new Error('the server has no tool that accepts this file type');
	}

	if (file.size > FILE_INTAKE_MAX_BYTES) {
		throw new Error(`larger than ${FILE_INTAKE_MAX_BYTES / (1024 * 1024)} MB`);
	}

	const result = await ToolsService.executeTool(
		tool,
		{ base64: await readFileAsBase64(file), name: file.name },
		undefined,
		currentCwd()
	);
	const firstLine = result.content.split('\n')[0];

	if (result.isError || !firstLine.startsWith(FILE_INTAKE_SAVED_PREFIX)) {
		throw new Error(result.content || 'the server did not save the file');
	}

	const path = firstLine.slice(FILE_INTAKE_SAVED_PREFIX.length).trim();
	const neededFonts = parseNeededFonts(result.content);

	if (neededFonts.length > 0) warnAboutFonts(path, file.name, neededFonts);

	return `${FILE_INTAKE_NOTE_PREFIX}${path}\nIts content is not included in this message. Use the available tools with that path to read or edit it.`;
}

/** Path of the file on the server, if this attachment text is a file intake note. */
export function getSavedFilePath(note: string | undefined): string | null {
	if (!note?.startsWith(FILE_INTAKE_NOTE_PREFIX)) return null;

	return note.slice(FILE_INTAKE_NOTE_PREFIX.length).split('\n')[0].trim() || null;
}

/** Download the file as it is on the server now, with any edits the model made. */
export async function downloadSavedFile(path: string): Promise<void> {
	try {
		const raw = await ToolsService.executeToolRaw(
			BuiltInTool.SERVER_READ_FILE,
			{ path },
			undefined,
			undefined,
			RESP_TYPE_BASE64
		);

		if (ToolResponseField.ERROR in raw) throw new Error(String(raw[ToolResponseField.ERROR]));

		const binary = atob(typeof raw.base64 === 'string' ? raw.base64 : '');
		const bytes = new Uint8Array(binary.length);

		for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

		const url = URL.createObjectURL(new Blob([bytes]));
		const a = document.createElement('a');

		a.href = url;
		a.download = path.split('/').pop() || 'download';
		document.body.appendChild(a);
		a.click();
		document.body.removeChild(a);
		URL.revokeObjectURL(url);
	} catch (err) {
		const reason = err instanceof Error ? err.message : String(err);

		// read_file is the only server tool that returns raw bytes
		toast.error(`Could not download the file (the server needs --tools read_file): ${reason}`, {
			duration: 8000
		});
	}
}
