/** Route table type shared by the mock handler modules. */
export type Handler = (match: RegExpMatchArray, query: URLSearchParams, body: unknown, now: number) => unknown;
/** [method, path regex (relative to /api/v1), handler, success status (default 200)] */
export type Route = [string, RegExp, Handler, number?];
