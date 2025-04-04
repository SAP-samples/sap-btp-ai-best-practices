@path    : 'orchestration'
@requires: 'authenticated-user'
service OrchestrationService {
  action askCapitalOfCountry(country : String) returns String;
}
