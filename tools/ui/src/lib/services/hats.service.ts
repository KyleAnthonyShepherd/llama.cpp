/**
 * HatsService - Stateless calls to the server's hats tool
 *
 * One server tool, found by the end of its name, takes an `action` and
 * answers JSON. No reactive state; consumed by hatsStore.
 */

import { ToolsService } from '$lib/services/tools.service';
import type {
	HatDetail,
	HatList,
	HatSaveFields,
	HatSummary,
	HatWear,
	ToolExecutionResult
} from '$lib/types';

const ERROR_PREFIX = 'ERROR: ';

function parse<T>(result: ToolExecutionResult): T {
	if (result.isError) {
		const text = result.content.startsWith(ERROR_PREFIX)
			? result.content.slice(ERROR_PREFIX.length)
			: result.content;

		throw new Error(text || 'the hats tool failed');
	}

	return JSON.parse(result.content) as T;
}

function readFileAsBase64(file: File): Promise<string> {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();

		reader.onload = () => resolve((reader.result as string).split(',')[1] ?? '');
		reader.onerror = () => reject(reader.error);
		reader.readAsDataURL(file);
	});
}

export class HatsService {
	static async build(tool: string, name: string): Promise<HatSummary> {
		return parse(await ToolsService.executeTool(tool, { action: 'build', name }));
	}

	static async cancel(tool: string, name: string): Promise<HatSummary> {
		return parse(await ToolsService.executeTool(tool, { action: 'cancel', name }));
	}

	static async get(tool: string, name: string): Promise<HatDetail> {
		return parse(await ToolsService.executeTool(tool, { action: 'get', name }));
	}

	static async list(tool: string): Promise<HatList> {
		return parse(await ToolsService.executeTool(tool, { action: 'list' }));
	}

	static async putDocument(tool: string, name: string, file: File): Promise<HatDetail> {
		return parse(
			await ToolsService.executeTool(tool, {
				action: 'put_document',
				base64: await readFileAsBase64(file),
				file: file.name,
				name
			})
		);
	}

	static async remove(tool: string, name: string): Promise<void> {
		parse(await ToolsService.executeTool(tool, { action: 'delete', name }));
	}

	static async removeDocument(tool: string, name: string, file: string): Promise<HatDetail> {
		return parse(await ToolsService.executeTool(tool, { action: 'remove_document', file, name }));
	}

	static async save(tool: string, name: string, fields: HatSaveFields): Promise<HatDetail> {
		return parse(await ToolsService.executeTool(tool, { ...fields, action: 'save', name }));
	}

	static async status(tool: string, name: string): Promise<HatSummary> {
		return parse(await ToolsService.executeTool(tool, { action: 'status', name }));
	}

	static async wear(tool: string, name: string, signal?: AbortSignal): Promise<HatWear> {
		return parse(await ToolsService.executeTool(tool, { action: 'wear', name }, signal));
	}
}
