using {cuid} from '@sap/cds/common';

context ai.db {
    entity Science_Data : cuid {
        Topic           : String;
        DifficultyLevel : String(100);
        Category        : String(100);
        Embedding       : Vector;
    }
}
