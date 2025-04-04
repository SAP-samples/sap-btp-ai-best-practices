@path    : 'orchestration'
@requires: 'any'
service OrchestrationService {
  action askCapitalOfCountry(country : String) returns String;
}
