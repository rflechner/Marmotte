namespace Marmotte

open System
open System.Reflection
open System.Text
open System.Text.Json
open System.Threading.Tasks
open Microsoft.AspNetCore.Builder
open Microsoft.AspNetCore.Http
open Microsoft.AspNetCore.Http.Json
open Microsoft.AspNetCore.Routing.Template
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Options
open Microsoft.FSharp.Core

[<RequireQualifiedAccess>]
module Restful =

    type ApiRequestHandler<'tin, 'tout, 'terr> = 'tin -> Task<Result<'tout, 'terr>>
    
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
    
    let getInputDto<'tin> (ctx: HttpContext) =
        task {
            let isBodyJson = ctx.Request.Headers.ContentType |> Seq.exists (fun v -> v.Equals ApplicationJson)
            if not isBodyJson
            then return Activator.CreateInstance<'tin>()
            else            
                let jsonOptions = ctx.RequestServices.GetRequiredService<IOptions<JsonOptions>>().Value.SerializerOptions
                return! JsonSerializer.DeserializeAsync<'tin>(ctx.Request.Body, jsonOptions)
        }
    
    let map<'tin, 'tout, 'terr> (verb: string) (routeTemplate: string) (handler: ApiRequestHandler<'tin, 'tout, 'terr>) (configure: RouteConfig -> RouteConfig) (ctx: ResourceContext) =
        let template = TemplateParser.Parse(routeTemplate)
        let routeParameters = template.Parameters |> Seq.map (fun p -> p.Name.ToLowerInvariant(), p) |> dict
        let modelType = typeof<'tin>
        let properties = modelType.GetProperties() |> Seq.map(fun p -> p.Name.ToLowerInvariant(), p) |> dict
        let func =
            Func<HttpContext, Task>(
                fun ctx ->
                    task {
                        let! dto = getInputDto<'tin> ctx
                        for prop in properties do
                            // Bind matching route parameter to property
                            match prop.Key.ToLowerInvariant() |> routeParameters.TryGetValue with
                            | false, _ -> ()
                            | true, param ->
                                match ctx.Request.RouteValues.TryGetValue param.Name with
                                | false, _ -> ()
                                | true, param ->
                                    let converted = Convert.ChangeType(param, prop.Value.PropertyType)
                                    prop.Value.SetValue(dto, converted)
                            
                            // Inject HttpContext in request DTO if wanted
                            if prop.Value.PropertyType = typeof<HttpContext>
                            then prop.Value.SetValue(dto, ctx)
                            // Inject from services
                            elif prop.Value.GetCustomAttribute<Microsoft.AspNetCore.Mvc.FromServicesAttribute>() <> null
                            then
                                let service = ctx.RequestServices.GetRequiredService(prop.Value.PropertyType)
                                prop.Value.SetValue(dto, service)

                        let! result = handler dto
                        match result with
                        | Error err ->
                            return! Results.InternalServerError(err).ExecuteAsync(ctx)
                        | Ok result ->
                            return! Results.Ok(result).ExecuteAsync(ctx)
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

    let patch<'tin, 'tout, 'terr> (routeTemplate: string) (handler: ApiRequestHandler<'tin, 'tout, 'terr>) =
        map HttpMethods.Patch routeTemplate handler

    let put<'tin, 'tout, 'terr> (routeTemplate: string) (handler: ApiRequestHandler<'tin, 'tout, 'terr>) =
        map HttpMethods.Put routeTemplate handler

    let trace<'tin, 'tout, 'terr> (routeTemplate: string) (handler: ApiRequestHandler<'tin, 'tout, 'terr>) =
        map HttpMethods.Trace routeTemplate handler

    let options<'tin, 'tout, 'terr> (routeTemplate: string) (handler: ApiRequestHandler<'tin, 'tout, 'terr>) =
        map HttpMethods.Options routeTemplate handler
    
    let addResource (name:string) (f: ResourceContext -> RouteHandlerBuilder) (app: WebApplication) =
        let ctx = { Name=name; Tags=name; App=app }
        f ctx

