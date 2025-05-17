// This is an automatically generated file. Please do not change its contents manually!
import * as __ from './../_/index.js';

export default class {
}

export function _ScienceDataAspect<TBase extends new (...args: any[]) => object>(Base: TBase) {
  return class ScienceData extends Base {
    declare ID?: __.Key<string>
    declare Topic?: string | null
    declare DifficultyLevel?: string | null
    declare Category?: string | null
    static readonly kind: 'entity' | 'type' | 'aspect' = 'entity';
    declare static readonly keys: __.KeysOf<ScienceData>;
    declare static readonly elements: __.ElementsOf<ScienceData>;
    declare static readonly actions: globalThis.Record<never, never>;
  };
}
/**
* Aspect for entities with canonical universal IDs
* 
* See https://cap.cloud.sap/docs/cds/common#aspect-cuid
*/
export class ScienceData extends _ScienceDataAspect(__.Entity) {}
Object.defineProperty(ScienceData, 'name', { value: 'RagService.ScienceData' })
Object.defineProperty(ScienceData, 'is_singular', { value: true })
/**
* Aspect for entities with canonical universal IDs
* 
* See https://cap.cloud.sap/docs/cds/common#aspect-cuid
*/
export class ScienceData_ extends Array<ScienceData> {$count?: number}
Object.defineProperty(ScienceData_, 'name', { value: 'RagService.ScienceData' })

export function _ScienceDataUploadAspect<TBase extends new (...args: any[]) => object>(Base: TBase) {
  return class ScienceDataUpload extends Base {
    declare content?: import("stream").Readable | null
    declare mediaType?: string | null
    static readonly kind: 'entity' | 'type' | 'aspect' = 'entity';
    declare static readonly keys: __.KeysOf<ScienceDataUpload>;
    declare static readonly elements: __.ElementsOf<ScienceDataUpload>;
    declare static readonly actions: globalThis.Record<never, never>;
  };
}
export class ScienceDataUpload extends _ScienceDataUploadAspect(__.Entity) {}
Object.defineProperty(ScienceDataUpload, 'name', { value: 'RagService.ScienceDataUpload' })
Object.defineProperty(ScienceDataUpload, 'is_singular', { value: true })
export class ScienceDataUpload_ extends Array<ScienceDataUpload> {$count?: number}
Object.defineProperty(ScienceDataUpload_, 'name', { value: 'RagService.ScienceDataUpload' })

export function _categoryListAspect<TBase extends new (...args: any[]) => object>(Base: TBase) {
  return class categoryList extends Base {
    declare Category?: __.Key<string>
    static readonly kind: 'entity' | 'type' | 'aspect' = 'entity';
    declare static readonly keys: __.KeysOf<categoryList>;
    declare static readonly elements: __.ElementsOf<categoryList>;
    declare static readonly actions: globalThis.Record<never, never>;
  };
}
/**
* Aspect for entities with canonical universal IDs
* 
* See https://cap.cloud.sap/docs/cds/common#aspect-cuid
*/
export class categoryList extends _categoryListAspect(__.Entity) {}
Object.defineProperty(categoryList, 'name', { value: 'RagService.categoryList' })
Object.defineProperty(categoryList, 'is_singular', { value: true })
/**
* Aspect for entities with canonical universal IDs
* 
* See https://cap.cloud.sap/docs/cds/common#aspect-cuid
*/
export class categoryList_ extends Array<categoryList> {$count?: number}
Object.defineProperty(categoryList_, 'name', { value: 'RagService.categoryList' })

export function _difficultyLevelListAspect<TBase extends new (...args: any[]) => object>(Base: TBase) {
  return class difficultyLevelList extends Base {
    declare DifficultyLevel?: __.Key<string>
    static readonly kind: 'entity' | 'type' | 'aspect' = 'entity';
    declare static readonly keys: __.KeysOf<difficultyLevelList>;
    declare static readonly elements: __.ElementsOf<difficultyLevelList>;
    declare static readonly actions: globalThis.Record<never, never>;
  };
}
/**
* Aspect for entities with canonical universal IDs
* 
* See https://cap.cloud.sap/docs/cds/common#aspect-cuid
*/
export class difficultyLevelList extends _difficultyLevelListAspect(__.Entity) {}
Object.defineProperty(difficultyLevelList, 'name', { value: 'RagService.difficultyLevelList' })
Object.defineProperty(difficultyLevelList, 'is_singular', { value: true })
/**
* Aspect for entities with canonical universal IDs
* 
* See https://cap.cloud.sap/docs/cds/common#aspect-cuid
*/
export class difficultyLevelList_ extends Array<difficultyLevelList> {$count?: number}
Object.defineProperty(difficultyLevelList_, 'name', { value: 'RagService.difficultyLevelList' })
