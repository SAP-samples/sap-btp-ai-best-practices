// This is an automatically generated file. Please do not change its contents manually!
import * as _ from './../../index.js';
import * as __ from './../../_/index.js';

export function _Science_DataAspect<TBase extends new (...args: any[]) => object>(Base: TBase) {
  return class Science_Data extends _._cuidAspect(Base) {
    declare Topic?: string | null
    declare DifficultyLevel?: string | null
    declare Category?: string | null
    declare Embedding?: string | null
    static override readonly kind: 'entity' | 'type' | 'aspect' = 'entity';
    declare static readonly keys: __.KeysOf<Science_Data> & typeof _.cuid.keys;
    declare static readonly elements: __.ElementsOf<Science_Data>;
    declare static readonly actions: typeof _.cuid.actions & globalThis.Record<never, never>;
  };
}
/**
* Aspect for entities with canonical universal IDs
* 
* See https://cap.cloud.sap/docs/cds/common#aspect-cuid
*/
export class Science_Data extends _Science_DataAspect(__.Entity) {}
Object.defineProperty(Science_Data, 'name', { value: 'ai.db.Science_Data' })
Object.defineProperty(Science_Data, 'is_singular', { value: true })
/**
* Aspect for entities with canonical universal IDs
* 
* See https://cap.cloud.sap/docs/cds/common#aspect-cuid
*/
export class Science_Data_ extends Array<Science_Data> {$count?: number}
Object.defineProperty(Science_Data_, 'name', { value: 'ai.db.Science_Data' })
