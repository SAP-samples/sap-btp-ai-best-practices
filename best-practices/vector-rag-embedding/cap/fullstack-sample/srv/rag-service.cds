using {ai.db as db} from '../db/schema';

@requires: 'any'
service RagService {
  entity ScienceData         as
    projection on db.Science_Data
    excluding {
      Embedding
    };

  @cds.persistence.skip
  @odata.singleton
  entity ScienceDataUpload {
    @Core.MediaType  : mediaType content : LargeBinary;
    @Core.IsMediaType: true mediaType    : String;
  };


  // Value Lists
  @readonly
  entity categoryList        as
    select from db.Science_Data distinct {
      key Category
    };

  @readonly
  entity difficultyLevelList as
    select from db.Science_Data distinct {
      key DifficultyLevel
    };
}
