// This is an automatically generated file. Please do not change its contents manually!

import { type } from '@sap/cds'

export type ElementsOf<T> = {[name in keyof Required<T>]: type }

export namespace Association {
    export type to <T> = T;
    export namespace to {
        export type many <T extends readonly any[]> = T;
    }
}

export namespace Composition {
    export type of <T> = T;
    export namespace of {
        export type many <T extends readonly any[]> = T;
    }
}

export class Entity {
    static data<T extends Entity> (this:T, _input:Object) : T {
        return {} as T // mock
    }
}

export type EntitySet<T> = T[] & {
    data (input:object[]) : T[]
    data (input:object) : T
};

export type DraftEntity<T> = T & {
    IsActiveEntity?: boolean | null
    HasActiveEntity?: boolean | null
    HasDraftEntity?: boolean | null
    DraftAdministrativeData_DraftUUID?: string | null
}

export type DraftOf<T> = { new(...args: any[]): DraftEntity<T> }
export type DraftsOf<T> = typeof Array<DraftEntity<T>>

export type DeepRequired<T> = { 
    [K in keyof T]: DeepRequired<Unkey<T[K]>>
} & Exclude<Required<T>, null>;

const key = Symbol('key')  // to avoid .key showing up in IDE's auto-completion
export type Key<T> = T & {[key]?: true}

export type KeysOf<T> = {
  [K in keyof T as NonNullable<T[K]> extends Key<unknown> ? K : never]-?: Key<{}>  // T[K] 
}

export type Unkey<T> = T extends Key<infer U> ? U : T

/**
 * Dates and timestamps are strings during runtime, so cds-typer represents them as such.
 */
export type CdsDate = `${number}${number}${number}${number}-${number}${number}-${number}${number}`;
/**
 * @see {@link CdsDate}
 */
export type CdsDateTime = string;
/**
 * @see {@link CdsDate}
 */
export type CdsTime = `${number}${number}:${number}${number}:${number}${number}`;
/**
 * @see {@link CdsDate}
 */
export type CdsTimestamp = string;

export type CdsMap = { [key: string]: unknown };
