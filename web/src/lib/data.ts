import { useEffect, useState } from "react";

const cache = new Map<string, Promise<any>>();

export function load<T = any>(path: string): Promise<T> {
  if (!cache.has(path)) {
    cache.set(path, fetch(`/data/${path}`).then((r) => {
      if (!r.ok) throw new Error(`${path}: ${r.status}`);
      return r.json();
    }));
  }
  return cache.get(path)!;
}

/** Fetch a static JSON file from /data once per session. Returns undefined while loading. */
export function useData<T = any>(path: string | null): { data: T | undefined; error?: string } {
  const [state, setState] = useState<{ data: T | undefined; error?: string }>({ data: undefined });
  useEffect(() => {
    let live = true;
    if (!path) return;
    setState({ data: undefined });
    load<T>(path).then((d) => live && setState({ data: d }), (e) => live && setState({ data: undefined, error: String(e) }));
    return () => { live = false; };
  }, [path]);
  return state;
}

export const bucketOf = (id: number) => `alerts/${id % 40}.json`;
