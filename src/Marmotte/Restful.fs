namespace Marmotte

open System
open System.Net.Http
open System.Threading.Tasks
open Microsoft.AspNetCore.Builder
open Microsoft.AspNetCore.Http
open Microsoft.AspNetCore.Routing
open Microsoft.AspNetCore.Routing.Template
open Microsoft.FSharp.Core

[<RequireQualifiedAccess>]
module Restful =

    type ApiRequestHandler<'tin, 'tout, 'terr> = HttpContext -> 'tin -> Task<Result<'tout, 'terr>>
    
    type ResourceContext =
        { Name: string
          Tags: string
          App: WebApplication }

    let map<'tin, 'tout, 'terr> (verb: string) (routeTemplate: string) (handler: ApiRequestHandler<'tin, 'tout, 'terr>) (ctx: ResourceContext) =
        let template = TemplateParser.Parse(routeTemplate)
        let func =
            Func<HttpContext, Task>(
                fun ctx ->
                    task {
                        let dto = Activator.CreateInstance<'tin>()
                        
                        for p in template.Parameters do
                            printfn $"Parameter {p.Name}"
                        
                        let! result = handler ctx dto
                        match result with
                        | Error err ->
                            return! Results.InternalServerError(err).ExecuteAsync(ctx)
                        | Ok result ->
                            return! Results.Ok(result).ExecuteAsync(ctx)
                    }
            )
        let routeBuilder = ctx.App.MapMethods(routeTemplate, [verb], func)
        routeBuilder.Produces<'tout>().Produces<'terr>(statusCode=500).WithName(ctx.Name).WithTags(ctx.Name)

    let get<'tin, 'tout, 'terr> (routeTemplate: string) (handler: ApiRequestHandler<'tin, 'tout, 'terr>) =
        map HttpMethods.Get routeTemplate handler
    
    let post<'tin, 'tout, 'terr> (routeTemplate: string) (handler: ApiRequestHandler<'tin, 'tout, 'terr>) =
        map HttpMethods.Post routeTemplate handler

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


