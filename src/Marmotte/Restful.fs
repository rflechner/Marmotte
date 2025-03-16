namespace Marmotte

open System
open System.Collections
open System.Collections.Generic
open System.Reflection
open System.Text
open System.Text.Json
open System.Threading.Tasks
open Microsoft.AspNetCore.Builder
open Microsoft.AspNetCore.Http
open Microsoft.AspNetCore.Http.Json
open Microsoft.AspNetCore.Mvc.ModelBinding
open Microsoft.AspNetCore.Routing.Template
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Options
open Microsoft.FSharp.Core

[<RequireQualifiedAccess>]
module Restful =

    type SemanticResult<'tok, 'terr> =
        | Ok of 'tok
        | Accepted
        | NoContent
        | Created
        | CreatedAt of Location:Uri * 'tok option
        | NotFound of 'terr option
        | Conflict of 'terr option
        | BadRequest
        | Unauthorized
        | Forbidden
        | Failure of 'terr
        | StatusCode of Code:int * Data:obj option
  
    type ApiRequestHandler<'tin, 'tout, 'terr> = 'tin -> Task<SemanticResult<'tout, 'terr>>
    
    type ResourceContext =
        { Name: string
          Tags: string
          App: WebApplication }
        
    type RouteConfig =
        { OperationName: string option }
    
    [<RequireQualifiedAccess>]
    module RouteConfig =
        let create() = { OperationName = None }

    let [<Literal>] ApplicationJson = "application/json"
    
    let getInputRequest<'tin> (ctx: HttpContext) =
        task {
            let isBodyJson = ctx.Request.Headers.ContentType |> Seq.exists (fun v -> v.Equals ApplicationJson)
            if not isBodyJson
            then return Activator.CreateInstance<'tin>()
            else            
                let jsonOptions = ctx.RequestServices.GetRequiredService<IOptions<JsonOptions>>().Value.SerializerOptions
                return! JsonSerializer.DeserializeAsync<'tin>(ctx.Request.Body, jsonOptions)
        }
    
    let bindRequest<'tin> (ctx:HttpContext) (properties: IDictionary<string, PropertyInfo>) (routeParameters: IDictionary<string, TemplatePart>) =
        task {
            let! request = getInputRequest<'tin> ctx

            for prop in properties do
                // Bind matching route parameter to property
                match prop.Key.ToLowerInvariant() |> routeParameters.TryGetValue with
                | false, _ -> ()
                | true, param ->
                    match ctx.Request.RouteValues.TryGetValue param.Name with
                    | false, _ -> ()
                    | true, param ->
                        let converted = Convert.ChangeType(param, prop.Value.PropertyType)
                        prop.Value.SetValue(request, converted)
                
                // Inject HttpContext in request DTO if wanted
                if prop.Value.PropertyType = typeof<HttpContext>
                then prop.Value.SetValue(request, ctx)
                // Inject from services
                elif prop.Value.GetCustomAttribute<Microsoft.AspNetCore.Mvc.FromServicesAttribute>() <> null
                then
                    let service = ctx.RequestServices.GetRequiredService(prop.Value.PropertyType)
                    prop.Value.SetValue(request, service)
                // Inject from query
                elif prop.Value.GetCustomAttribute<Microsoft.AspNetCore.Mvc.FromQueryAttribute>() <> null
                then
                    let attr = prop.Value.GetCustomAttribute<Microsoft.AspNetCore.Mvc.FromQueryAttribute>()
                    let name = if String.IsNullOrWhiteSpace attr.Name then prop.Value.Name else attr.Name
                    match ctx.Request.Query.TryGetValue name with
                    | true, values when values.Count > 0 ->
                        let value = values.Item 0 
                        let converted = Convert.ChangeType(value, prop.Value.PropertyType)
                        prop.Value.SetValue(request, converted)
                    | _ -> ()
            
            return request
        }
        
    let mapResult result =
        match result with
        | Ok result -> Results.Ok result
        | Accepted -> Results.Accepted(value=result)
        | NoContent -> Results.NoContent()
        | Created -> Results.Created()
        | CreatedAt(location, None) -> Results.Created(location, null)
        | CreatedAt(location, Some data) -> Results.Created(location, value=data)
        | NotFound None -> Results.NotFound()
        | NotFound (Some data) -> Results.NotFound(data)
        | Conflict None -> Results.Conflict()
        | Conflict (Some data) -> Results.Conflict(data)
        | BadRequest -> Results.BadRequest()
        | Unauthorized -> Results.Unauthorized()
        | Forbidden -> Results.Forbid()
        | Failure err -> Results.InternalServerError err
        | StatusCode(code, None) -> Results.StatusCode code
        | StatusCode(code, Some data) ->
            Http.FunResult(
                fun ctx ->
                    task {
                        ctx.Response.StatusCode <- code
                        do! HttpResponseJsonExtensions.WriteAsJsonAsync(ctx.Response, data)
                    }
                )
    
    let map<'tin, 'tout, 'terr> (verb: string) (routeTemplate: string) (handler: ApiRequestHandler<'tin, 'tout, 'terr>) (configure: RouteConfig -> RouteConfig) (ctx: ResourceContext) =
        let template = TemplateParser.Parse(routeTemplate)
        let routeParameters = template.Parameters |> Seq.map (fun p -> p.Name.ToLowerInvariant(), p) |> dict
        let modelType = typeof<'tin>
        let properties = modelType.GetProperties() |> Seq.map(fun p -> p.Name.ToLowerInvariant(), p) |> dict
        let func =
            Func<HttpContext, Task>(
                fun ctx ->
                    task {
                        let! request = bindRequest ctx properties routeParameters
                        let! restfulResult = handler request
                        let result = mapResult restfulResult
                        return! result.ExecuteAsync ctx
                    }
            )
        let sb = StringBuilder()
        '/' |> sb.Append |> ignore
        
        if ctx.Name |> String.IsNullOrWhiteSpace |> not && ctx.Name <> "/"
        then
            ctx.Name.ToLowerInvariant().TrimEnd '/' |> sb.Append |> ignore
            '/' |> sb.Append |> ignore
        
        routeTemplate.TrimStart '/' |> sb.Append |> ignore
        
        let config = RouteConfig.create() |> configure
        let operationName = config.OperationName |> Option.defaultWith (fun () -> $"{Helpers.ucFirst verb}{ctx.Name}")
        
        let routeBuilder = ctx.App.MapMethods(sb.ToString(), [verb], func)
        routeBuilder.Produces<'tout>().Produces<'terr>(statusCode=500).WithName(operationName).WithTags(ctx.Tags)

    let get<'tin, 'tout, 'terr> (routeTemplate: string) (handler: ApiRequestHandler<'tin, 'tout, 'terr>) =
        map HttpMethods.Get routeTemplate handler
    
    let post<'tin, 'tout, 'terr> (routeTemplate: string) (handler: ApiRequestHandler<'tin, 'tout, 'terr>) (configure: RouteConfig -> RouteConfig) (ctx: ResourceContext) =
        let route = map HttpMethods.Post routeTemplate handler configure ctx
        route.Accepts<'tin> ApplicationJson

    let delete<'tin, 'tout, 'terr> (routeTemplate: string) (handler: ApiRequestHandler<'tin, 'tout, 'terr>) =
        map HttpMethods.Delete routeTemplate handler

    let patch<'tin, 'tout, 'terr> (routeTemplate: string) (handler: ApiRequestHandler<'tin, 'tout, 'terr>) configure ctx =
        let route = map HttpMethods.Patch routeTemplate handler configure ctx
        route.Accepts<'tin> ApplicationJson

    let put<'tin, 'tout, 'terr> (routeTemplate: string) (handler: ApiRequestHandler<'tin, 'tout, 'terr>) configure ctx =
        let route = map HttpMethods.Put routeTemplate handler configure ctx
        route.Accepts<'tin> ApplicationJson

    let trace<'tin, 'tout, 'terr> (routeTemplate: string) (handler: ApiRequestHandler<'tin, 'tout, 'terr>) =
        map HttpMethods.Trace routeTemplate handler

    let options<'tin, 'tout, 'terr> (routeTemplate: string) (handler: ApiRequestHandler<'tin, 'tout, 'terr>) =
        map HttpMethods.Options routeTemplate handler
    
    let addResource (name:string) (f: ResourceContext -> RouteHandlerBuilder) (app: WebApplication) =
        let ctx = { Name=name; Tags=name; App=app }
        f ctx

