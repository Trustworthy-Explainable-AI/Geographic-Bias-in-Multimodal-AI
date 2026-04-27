import fs from "fs";
import path from "path";

export type EvalItem = {
    task_id: string;
    image_id: string;
    image_filename: string;
    image_url: string;
    model: string;
    region: string;
    income_quintile: string;
    ground_truth: string;
    predicted: string;
    model_error_type: string;
};

export type EvalResponse = {
    taskId: string;
    sessionId: string;
    verdict: "true" | "false" | "unsure";
    note: string;
    createdAt: string;
};

export type RuntimeState = {
    targetSampleSize: number;
    sampledTaskIds: string[];
    responses: EvalResponse[];
    skipped: {
        sessionId: string;
        taskId: string;
        createdAt: string;
    }[];
};

const DATA_DIR = path.join(process.cwd(), "data");
const ITEMS_PATH = path.join(DATA_DIR, "items.json");
const RUNTIME_PATH = path.join(DATA_DIR, "runtime.json");
const DEFAULT_SAMPLE_SIZE = 120;

const SUPABASE_URL = process.env.SUPABASE_URL?.replace(/\/$/, "") ?? "";
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY ?? "";

const RUNTIME_TABLE = process.env.SUPABASE_RUNTIME_TABLE ?? "eval_runtime";
const RESPONSES_TABLE = process.env.SUPABASE_RESPONSES_TABLE ?? "eval_responses";
const SKIPPED_TABLE = process.env.SUPABASE_SKIPPED_TABLE ?? "eval_skipped";

function hasRemoteStore(): boolean {
    return Boolean(SUPABASE_URL && SUPABASE_SERVICE_ROLE_KEY);
}

async function supabaseFetch(pathname: string, init?: RequestInit): Promise<Response> {
    const response = await fetch(`${SUPABASE_URL}${pathname}`, {
        ...init,
        headers: {
            "Content-Type": "application/json",
            apikey: SUPABASE_SERVICE_ROLE_KEY,
            Authorization: `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
            ...(init?.headers ?? {})
        },
        cache: "no-store"
    });

    if (!response.ok) {
        const text = await response.text();
        throw new Error(`Supabase request failed: ${response.status} ${text}`);
    }

    return response;
}

async function getOrCreateRemoteRuntimeConfig(): Promise<{ targetSampleSize: number; sampledTaskIds: string[] }> {
    const selectPath = `/rest/v1/${RUNTIME_TABLE}?id=eq.1&select=target_sample_size,sampled_task_ids`;
    const selectResponse = await supabaseFetch(selectPath);
    const rows = (await selectResponse.json()) as {
        target_sample_size?: number;
        sampled_task_ids?: string[];
    }[];

    if (rows.length > 0) {
        return {
            targetSampleSize: rows[0].target_sample_size ?? DEFAULT_SAMPLE_SIZE,
            sampledTaskIds: Array.isArray(rows[0].sampled_task_ids) ? rows[0].sampled_task_ids : []
        };
    }

    const defaultRow = [{ id: 1, target_sample_size: DEFAULT_SAMPLE_SIZE, sampled_task_ids: [] as string[] }];
    await supabaseFetch(`/rest/v1/${RUNTIME_TABLE}?on_conflict=id`, {
        method: "POST",
        headers: { Prefer: "resolution=merge-duplicates,return=minimal" },
        body: JSON.stringify(defaultRow)
    });

    return { targetSampleSize: DEFAULT_SAMPLE_SIZE, sampledTaskIds: [] };
}

async function readLocalRuntime(): Promise<RuntimeState> {
    const raw = JSON.parse(fs.readFileSync(RUNTIME_PATH, "utf-8")) as Partial<RuntimeState>;
    return {
        targetSampleSize: raw.targetSampleSize ?? DEFAULT_SAMPLE_SIZE,
        sampledTaskIds: raw.sampledTaskIds ?? [],
        responses: raw.responses ?? [],
        skipped: raw.skipped ?? []
    };
}

export function readItems(): EvalItem[] {
    return JSON.parse(fs.readFileSync(ITEMS_PATH, "utf-8")) as EvalItem[];
}

export async function readRuntime(): Promise<RuntimeState> {
    if (!hasRemoteStore()) {
        return readLocalRuntime();
    }

    const runtimeConfig = await getOrCreateRemoteRuntimeConfig();

    const [responsesResponse, skippedResponse] = await Promise.all([
        supabaseFetch(`/rest/v1/${RESPONSES_TABLE}?select=task_id,session_id,verdict,note,created_at&limit=100000`),
        supabaseFetch(`/rest/v1/${SKIPPED_TABLE}?select=session_id,task_id,created_at&limit=100000`)
    ]);

    const responseRows = (await responsesResponse.json()) as {
        task_id: string;
        session_id: string;
        verdict: "true" | "false" | "unsure";
        note: string | null;
        created_at: string;
    }[];

    const skippedRows = (await skippedResponse.json()) as {
        session_id: string;
        task_id: string;
        created_at: string;
    }[];

    return {
        targetSampleSize: runtimeConfig.targetSampleSize,
        sampledTaskIds: runtimeConfig.sampledTaskIds,
        responses: responseRows.map((row) => ({
            taskId: row.task_id,
            sessionId: row.session_id,
            verdict: row.verdict,
            note: row.note ?? "",
            createdAt: row.created_at
        })),
        skipped: skippedRows.map((row) => ({
            sessionId: row.session_id,
            taskId: row.task_id,
            createdAt: row.created_at
        }))
    };
}

export async function writeRuntime(state: RuntimeState): Promise<void> {
    if (hasRemoteStore()) {
        await supabaseFetch(`/rest/v1/${RUNTIME_TABLE}?id=eq.1`, {
            method: "PATCH",
            headers: { Prefer: "return=minimal" },
            body: JSON.stringify({
                target_sample_size: state.targetSampleSize,
                sampled_task_ids: state.sampledTaskIds
            })
        });
        return;
    }

    fs.writeFileSync(RUNTIME_PATH, JSON.stringify(state, null, 2), "utf-8");
}

export async function reseedSample(targetSampleSize: number): Promise<RuntimeState> {
    const items = readItems();
    const state = await readRuntime();
    const chosen = items.slice(0, Math.min(targetSampleSize, items.length));
    const chosenSet = new Set(chosen.map((item) => item.task_id));

    state.targetSampleSize = targetSampleSize;
    state.sampledTaskIds = Array.from(chosenSet);
    state.responses = state.responses.filter((resp) => chosenSet.has(resp.taskId));
    state.skipped = state.skipped.filter((skip) => chosenSet.has(skip.taskId));

    await writeRuntime(state);

    if (hasRemoteStore()) {
        if (state.sampledTaskIds.length === 0) {
            await Promise.all([
                supabaseFetch(`/rest/v1/${RESPONSES_TABLE}`, { method: "DELETE" }),
                supabaseFetch(`/rest/v1/${SKIPPED_TABLE}`, { method: "DELETE" })
            ]);
        } else {
            const notInExpr = `not.in.(${state.sampledTaskIds.join(",")})`;
            const params = new URLSearchParams({ task_id: notInExpr });
            await Promise.all([
                supabaseFetch(`/rest/v1/${RESPONSES_TABLE}?${params.toString()}`, { method: "DELETE" }),
                supabaseFetch(`/rest/v1/${SKIPPED_TABLE}?${params.toString()}`, { method: "DELETE" })
            ]);
        }
    }

    return state;
}

export async function nextTaskForSession(sessionId: string): Promise<EvalItem | null> {
    const items = readItems();
    const runtimeConfig = hasRemoteStore()
        ? await getOrCreateRemoteRuntimeConfig()
        : await readRuntime();

    const sampledTaskIds = runtimeConfig.sampledTaskIds;

    let completed = new Set<string>();
    let skipped = new Set<string>();

    if (hasRemoteStore()) {
        const [completedResponse, skippedResponse] = await Promise.all([
            supabaseFetch(`/rest/v1/${RESPONSES_TABLE}?session_id=eq.${encodeURIComponent(sessionId)}&select=task_id`),
            supabaseFetch(`/rest/v1/${SKIPPED_TABLE}?session_id=eq.${encodeURIComponent(sessionId)}&select=task_id`)
        ]);

        const completedRows = (await completedResponse.json()) as { task_id: string }[];
        const skippedRows = (await skippedResponse.json()) as { task_id: string }[];
        completed = new Set(completedRows.map((row) => row.task_id));
        skipped = new Set(skippedRows.map((row) => row.task_id));
    } else {
        const state = runtimeConfig as RuntimeState;
        completed = new Set(
            state.responses.filter((r) => r.sessionId === sessionId).map((r) => r.taskId)
        );
        skipped = new Set(
            state.skipped.filter((s) => s.sessionId === sessionId).map((s) => s.taskId)
        );
    }

    const remaining = sampledTaskIds.filter((id) => !completed.has(id) && !skipped.has(id));
    if (remaining.length === 0) {
        return null;
    }

    const taskId = remaining[0];
    return items.find((item) => item.task_id === taskId) ?? null;
}

export async function submitTaskForSession(payload: {
    taskId: string;
    sessionId: string;
    verdict: "true" | "false" | "unsure";
    note?: string;
}): Promise<{ ok: true; duplicate: boolean }> {
    const { taskId, sessionId, verdict, note } = payload;

    if (hasRemoteStore()) {
        const duplicateParams = new URLSearchParams({
            task_id: `eq.${taskId}`,
            session_id: `eq.${sessionId}`,
            select: "task_id",
            limit: "1"
        });
        const duplicateResponse = await supabaseFetch(`/rest/v1/${RESPONSES_TABLE}?${duplicateParams.toString()}`);
        const duplicateRows = (await duplicateResponse.json()) as { task_id: string }[];
        if (duplicateRows.length > 0) {
            return { ok: true, duplicate: true };
        }

        await supabaseFetch(`/rest/v1/${RESPONSES_TABLE}`, {
            method: "POST",
            headers: { Prefer: "return=minimal" },
            body: JSON.stringify([{
                task_id: taskId,
                session_id: sessionId,
                verdict,
                note: note ?? "",
                created_at: new Date().toISOString()
            }])
        });
        return { ok: true, duplicate: false };
    }

    const runtime = await readLocalRuntime();
    const alreadyDone = runtime.responses.some(
        (response) => response.taskId === taskId && response.sessionId === sessionId
    );

    if (alreadyDone) {
        return { ok: true, duplicate: true };
    }

    runtime.responses.push({
        taskId,
        sessionId,
        verdict,
        note: note ?? "",
        createdAt: new Date().toISOString()
    });

    fs.writeFileSync(RUNTIME_PATH, JSON.stringify(runtime, null, 2), "utf-8");
    return { ok: true, duplicate: false };
}

export async function skipTaskForSession(sessionId: string, taskId: string): Promise<void> {
    if (hasRemoteStore()) {
        const checkAlreadyAnsweredParams = new URLSearchParams({
            task_id: `eq.${taskId}`,
            session_id: `eq.${sessionId}`,
            select: "task_id",
            limit: "1"
        });
        const [alreadySkippedResponse, alreadyAnsweredResponse] = await Promise.all([
            supabaseFetch(`/rest/v1/${SKIPPED_TABLE}?${checkAlreadyAnsweredParams.toString()}`),
            supabaseFetch(`/rest/v1/${RESPONSES_TABLE}?${checkAlreadyAnsweredParams.toString()}`)
        ]);

        const alreadySkippedRows = (await alreadySkippedResponse.json()) as { task_id: string }[];
        const alreadyAnsweredRows = (await alreadyAnsweredResponse.json()) as { task_id: string }[];

        if (alreadySkippedRows.length === 0 && alreadyAnsweredRows.length === 0) {
            await supabaseFetch(`/rest/v1/${SKIPPED_TABLE}`, {
                method: "POST",
                headers: { Prefer: "return=minimal" },
                body: JSON.stringify([{
                    task_id: taskId,
                    session_id: sessionId,
                    created_at: new Date().toISOString()
                }])
            });
        }
        return;
    }

    const state = await readLocalRuntime();

    const alreadySkipped = state.skipped.some(
        (skip) => skip.sessionId === sessionId && skip.taskId === taskId
    );
    const alreadyAnswered = state.responses.some(
        (resp) => resp.sessionId === sessionId && resp.taskId === taskId
    );

    if (!alreadySkipped && !alreadyAnswered) {
        state.skipped.push({
            sessionId,
            taskId,
            createdAt: new Date().toISOString()
        });
        fs.writeFileSync(RUNTIME_PATH, JSON.stringify(state, null, 2), "utf-8");
    }
}
