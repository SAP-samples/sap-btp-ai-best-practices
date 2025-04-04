@path    : 'orchestration'
@requires: 'any'
service OrchestrationService {
  action generateEmail(prompt : String, anonymize : Boolean) returns String;
}
